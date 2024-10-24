/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2015-2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2023 Martin Uecker
 * 2014-2016 Frank Ong
 * 2014-2018 Jon Tamir
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops_p.h"
#include "num/ops.h"
#include "num/mpi_ops.h"
#include "num/delayed.h"

#include "iter/misc.h"
#include "iter/monitor.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"
#include "linops/realval.h"

#include "noncart/nufft.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/mri2.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "motion/displacement.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "num/iovec.h"
#include "num/ops.h"
#include "num/vptr.h"

static const char help_str[] = "Parallel-imaging compressed-sensing reconstruction.\n";


static const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps, const long ksp_dims[DIMS],
			const long traj_dims[DIMS], const complex float* traj, struct nufft_conf_s conf,
			const long wgs_dims[DIMS], const complex float* weights,
			const long basis_dims[DIMS], const complex float* basis,
			const struct linop_s** fft_opp, unsigned long shared_img_dims)
{
	long coilim_dims[DIMS];
	long img_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG & ~shared_img_dims, img_dims, max_dims);

	long ksp_dims2[DIMS];
	md_copy_dims(DIMS, ksp_dims2, ksp_dims);
	ksp_dims2[COEFF_DIM] = max_dims[COEFF_DIM];

	debug_print_dims(DP_INFO, DIMS, ksp_dims2);
	debug_print_dims(DP_INFO, DIMS, coilim_dims);

	long map_strs[DIMS];
	md_calc_strides(DIMS, map_strs, map_dims, CFL_SIZE);


	const struct linop_s* nufft_op = nufft_create2(DIMS, ksp_dims2, coilim_dims,
						traj_dims, traj,
						(weights ? wgs_dims : NULL), weights,
						(basis ? basis_dims : NULL), basis, conf);

	const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims, img_dims, maps);
	const struct linop_s* lop = linop_chain(maps_op, nufft_op);
	linop_free(maps_op);

	if (NULL != fft_opp)
		*fft_opp = linop_clone(nufft_op);

	linop_free(nufft_op);

	return lop;
}


int main_pics(int argc, char* argv[argc])
{
	const char* ksp_file = NULL;
	const char* sens_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INFILE(true, &sens_file, "sensitivities"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	// Initialize default parameters

	struct sense_conf conf = sense_defaults;

	float bpsense_eps = -1.;

	int shift_mode = 0;
	bool randshift = true;
	bool overlapping_blocks = false;
	int maxiter = 30;
	float step = -1.;

	// Start time count

	double start_time = timestamp();

	// Read input options
	nufft_conf_options.toeplitz = true;
	nufft_conf_options.lowmem = false;

	const char* pat_file = NULL;
	const char* traj_file = NULL;
	const char* motion_file = NULL;
	const char* psf_ifile = NULL;
	const char* psf_ofile = NULL;

	float restrict_fov = -1.;
	bool scale_im = false;
	bool eigen = false;
	float scaling = 0.;

        // Simultaneous Multi-Slice
        bool sms = false;

	int llr_blk = 8;
	const char* wtype_str = "dau2";

	const char* image_truth_file = NULL;
	const char* image_start_file = NULL;

	const char* basis_file = NULL;

	struct admm_conf admm = { false, false, false, iter_admm_defaults.rho, iter_admm_defaults.maxitercg, false };
	struct fista_conf fista = { { -1., -1., -1. }, false };
	struct pridu_conf pridu = { 1., false, 0. };

	enum algo_t algo = ALGO_DEFAULT;

	bool hogwild = false;

	bool gpu_gridding = false;

	struct opt_reg_s ropts;
	opt_reg_init(&ropts);

	unsigned long loop_flags = 0UL;
	unsigned long shared_img_flags = 0UL;
	bool lowmem = false;

	unsigned long mpi_flags = 0UL;


	const struct opt_s opts[] = {

		{ 'l', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "\b1/-l2", "  toggle l1-wavelet or l2 regularization." },
		OPT_FLOAT('r', &ropts.lambda, "lambda", "regularization parameter"),
		{ 'R', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "<T>:A:B:C", "generalized regularization options (-Rh for help)" },
		OPT_SET('c', &conf.rvc, "real-value constraint"),
		OPT_FLOAT('s', &step, "step", "iteration stepsize"),
		OPT_PINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_INFILE('t', &traj_file, "file", "k-space trajectory"),
		OPT_CLEAR('n', &randshift, "disable random wavelet cycle spinning"),
		OPT_SET('N', &overlapping_blocks, "do fully overlapping LLR blocks"),
		OPT_SET('g', &bart_use_gpu, "use GPU"),
		OPTL_SET(0, "gpu-gridding", &gpu_gridding, "use GPU for gridding"),
		OPT_INFILE('p', &pat_file, "file", "pattern or weights"),
		OPTL_SET(0, "precond", &(conf.precond), "interpret weights as preconditioner"),
		OPT_PINT('b', &llr_blk, "blk", "Lowrank block size"),
		OPT_SET('e', &eigen, "Scale stepsize based on max. eigenvalue"),
		OPTL_SET(0, "adaptive-stepsize", &(pridu.adaptive_stepsize), "PRIDU adaptive step size"),
		OPTL_SET(0, "asl", &(ropts.asl), "ASL reconstruction"),
		OPTL_SET(0, "teasl", &(ropts.teasl), "Time-encoded ASL reconstruction"),
		OPTL_FLVEC2(0, "theta", &ropts.theta, "theta1:theta2", "PWI weight for ASL reconstruction"),
		OPTL_FLVECN(0, "tvscales", ropts.tvscales, "Scaling of derivatives in TV or TGV regularization"),
		OPTL_FLVECN(0, "tvscales2", ropts.tvscales2, "Scaling of secondary derivatives in ICTV reconstruction"),
		OPTL_FLVEC2(0, "alpha", &ropts.alpha, "alpha1:alpha0", "regularization parameter for TGV and ICTGV reconstruction"),
		OPTL_FLVEC2(0, "gamma", &ropts.gamma, "gamma1:gamma2", "regularization parameter for ICTV and ICTGV reconstruction"),
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_SET('D', &admm.dynamic_rho, "(ADMM dynamic step size)"),
		OPT_SET('F', &admm.fast, "(fast)"),
		OPT_SET('J', &admm.relative_norm, "(ADMM residual balancing)"),
		OPT_INFILE('T', &image_truth_file, "file", "(truth file)"),
		OPT_INFILE('W', &image_start_file, "<img>", "Warm start with <img>"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_INT('O', &conf.rwiter, "rwiter", "(reweighting)"),
		OPT_FLOAT('o', &conf.gamma, "gamma", "(reweighting)"),
		OPT_FLOAT('u', &admm.rho, "rho", "ADMM rho"),
		OPT_PINT('C', &admm.maxitercg, "iter", "ADMM max. CG iterations"),
		OPT_FLOAT('q', &conf.cclambda, "cclambda", "(cclambda)"),
		OPT_FLOAT('f', &restrict_fov, "rfov", "restrict FOV"),
		OPTL_SELECT('I', "ist", enum algo_t, &algo, ALGO_IST, "select IST"),
		OPTL_SELECT(0, "fista", enum algo_t, &algo, ALGO_FISTA, "select FISTA"),
		OPTL_SELECT(0, "eulermaruyama", enum algo_t, &algo, ALGO_EULERMARUYAMA, "select Euler Maruyama"),
		OPTL_SELECT('m', "admm", enum algo_t, &algo, ALGO_ADMM, "select ADMM"),
		OPTL_SELECT('a', "pridu", enum algo_t, &algo, ALGO_PRIDU, "select Primal Dual"),
		OPT_FLOAT('w', &scaling, "", "inverse scaling of the data"),
		OPT_SET('S', &scale_im, "re-scale the image after reconstruction"),
		OPT_ULONG('L', &loop_flags, "flags", "(batch-mode)"),
		OPTL_ULONG(0, "shared-img-dims", &shared_img_flags, "flags", "deselect image dims with flags"),
		OPT_SET('K', &nufft_conf_options.pcycle, "randshift for NUFFT"),
		OPT_INFILE('B', &basis_file, "file", "temporal (or other) basis"),
		OPT_FLOAT('P', &bpsense_eps, "eps", "Basis Pursuit formulation, || y- Ax ||_2 <= eps"),
		OPT_SET('M', &sms, "Simultaneous Multi-Slice reconstruction"),
		OPTL_SET('U', "lowmem", &lowmem, "Use low-mem mode of the nuFFT"),
		OPTL_CLEAR(0, "no-toeplitz", &nufft_conf_options.toeplitz, "(Turn off Toeplitz mode of nuFFT)"),
		OPTL_OUTFILE(0, "psf_export", &psf_ofile, "file", "Export PSF to file"),
		OPTL_INFILE(0, "psf_import", &psf_ifile, "file", "Import PSF from file"),
		OPTL_STRING(0, "wavelet", &wtype_str, "name", "wavelet type (haar,dau2,cdf44)"),
		OPTL_ULONG(0, "mpi", &mpi_flags, "flags", "(distribute over this dimensions with use of MPI)"),
		OPTL_FLVEC3(0, "fista_pqr", &fista.params, "p:q:r", "parameters for FISTA acceleration"),
		OPTL_SET(0, "fista_last", &fista.last, "end iteration with call to data consistency"),
		OPTL_INFILE(0, "motion-field", &motion_file, "file", "motion field"),
		OPTL_SUBOPT(0, "nufft-conf", "...", "configure nufft", N_nufft_conf_opts, nufft_conf_opts),
	};


	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (0 != mpi_flags)
		error("MPI is now supported via the BART --md-split-mpi-dims option!\n");

	if (0 != loop_flags)
		error("Looping only supported via BART generic looping interface!\n");

	if (0 <= bpsense_eps)
		conf.bpsense = true;

	admm.dynamic_tau = admm.relative_norm;

	struct nufft_conf_s nuconf = nufft_conf_options;

	if (conf.bpsense)
		nuconf.toeplitz = false;

	num_init_delayed();

	if (0 != bart_delayed_loop_flags && NULL != traj_file) {

		num_delayed_add_loop_dims(16);	// loop over decomposed phases in nuFFT
		num_delayed_add_loop_dims(17);	// loop over decomposed phases in nuFFT when computing PSF
	}

	if (lowmem) {

		assert(NULL != traj_file);

		num_delayed_add_loop_dims(16);	// loop over decomposed phases in nuFFT
		num_delayed_add_loop_dims(17);	// loop over decomposed phases in nuFFT when computing PSF
		num_init_delayed();
	}


	long max_dims[DIMS];
	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long coilim_dims[DIMS];
	long ksp_dims[DIMS];
	long traj_dims[DIMS];


	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);
	struct vptr_hint_s* hint = (0 != bart_mpi_split_flags) || bart_delayed_computations ? vptr_hint_create(bart_mpi_split_flags, DIMS, ksp_dims, bart_delayed_loop_flags) : NULL;
	kspace = vptr_wrap_cfl(DIMS, ksp_dims, CFL_SIZE, kspace, hint, true, false);

        if (sms) {

		if (NULL == traj_file)
			error("SMS is only supported for non-Cartesian trajectories.\n");

		nuconf.cfft |= SLICE_FLAG;

                debug_printf(DP_INFO, "SMS reconstruction: MB = %ld\n", ksp_dims[SLICE_DIM]);
        }

	if (ropts.asl && ropts.teasl)
		error("Use either TE-ASL or ASL reconstruction.\n");

	if (ropts.asl && 2 != ksp_dims[ITER_DIM])
		error("ASL reconstruction requires two slices (label and control) along ITER_DIM.\n");

	complex float* maps = load_cfl_wrap(sens_file, DIMS, map_dims, hint);

	unsigned long map_flags = md_nontriv_dims(DIMS, map_dims);

	map_flags |= FFT_FLAGS | SENS_FLAGS;



	long basis_dims[DIMS] = { }; // analyzer false positive
	complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl_wrap(basis_file, DIMS, basis_dims, hint);
		assert(!MD_IS_SET(bart_mpi_split_flags, TE_DIM));
		assert(!md_check_dimensions(DIMS, basis_dims, COEFF_FLAG | TE_FLAG));
	}

	long motion_dims[DIMS] = { };
	complex float* motion = NULL;
	unsigned long motion_flags = 0;

	if (NULL != motion_file) {

		motion = load_cfl_wrap(motion_file, DIMS, motion_dims, hint);
		assert(!MD_IS_SET(bart_mpi_split_flags, MOTION_DIM));
		assert(1 < motion_dims[MOTION_DIM]);
		motion_flags = md_nontriv_dims(DIMS, motion_dims) & ~MOTION_FLAG;
	}


	complex float* traj = NULL;

	if (NULL != traj_file)
		traj = load_cfl_wrap(traj_file, DIMS, traj_dims, hint);

	md_copy_dims(DIMS, max_dims, ksp_dims);
	md_copy_dims(5, max_dims, map_dims);

	long bmx_dims[DIMS];

	if (NULL != basis_file) {

		assert(1 == ksp_dims[COEFF_DIM]);

		assert(basis_dims[TE_DIM] == ksp_dims[TE_DIM]);

		max_dims[COEFF_DIM] = basis_dims[COEFF_DIM];

		md_select_dims(DIMS, ~MAPS_FLAG, bmx_dims, max_dims);
		debug_printf(DP_INFO, "Basis: ");
		debug_print_dims(DP_INFO, DIMS, bmx_dims);

		max_dims[TE_DIM] = 1;

		debug_printf(DP_INFO, "Max:   ");
		debug_print_dims(DP_INFO, DIMS, max_dims);
	}

	md_select_dims(DIMS, ~COIL_FLAG & ~shared_img_flags, img_dims, max_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);

	if (!md_check_compat(DIMS, ~(MD_BIT(MAPS_DIM)|FFT_FLAGS), img_dims, map_dims))
		error("Dimensions of image and sensitivities do not match!\n");

	if ((NULL != traj_file) && (!md_check_compat(DIMS, ~0UL, ksp_dims, traj_dims)))
		error("Dimensions of data and trajectory do not match!\n");

	if ((NULL == traj_file) && (NULL != psf_ofile))
		error("Export of PSF for Cartesian scan requested!\n");


	assert(1 == ksp_dims[MAPS_DIM]);

	num_init_gpu_support();
	conf.gpu = bart_use_gpu;

	// print options

	if (0 != bart_mpi_split_flags)
		debug_printf(DP_INFO, "MPI reconstruction\n");

	if (conf.gpu)
		debug_printf(DP_INFO, "GPU reconstruction\n");

	if (map_dims[MAPS_DIM] > 1)
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", map_dims[MAPS_DIM]);

	if (conf.bpsense)
		debug_printf(DP_INFO, "Basis Pursuit formulation\n");

	if (hogwild)
		debug_printf(DP_INFO, "Hogwild stepsize\n");

	if (admm.dynamic_rho)
		debug_printf(DP_INFO, "ADMM Dynamic stepsize\n");

	if (admm.relative_norm)
		debug_printf(DP_INFO, "ADMM residual balancing\n");

	if (randshift)
		shift_mode = 1;

	if (overlapping_blocks) {

		if (randshift)
			debug_printf(DP_WARN, "Turning off random shifts\n");

		shift_mode = 2;

		debug_printf(DP_INFO, "Fully overlapping LLR blocks\n");
	}



	assert(!((conf.rwiter > 1) && (nuconf.toeplitz || conf.bpsense)));


	// initialize sampling pattern

	complex float* pattern = NULL;

	if (NULL != pat_file) {

		pattern = load_cfl_wrap(pat_file, DIMS, pat_dims, hint);

		assert(md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims));

	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = md_alloc_sameplace(DIMS, pat_dims, CFL_SIZE, kspace);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}


	long ksp_strs[DIMS];
	long pat_strs[DIMS];

	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

	md_zmul2(DIMS, ksp_dims, ksp_strs, kspace, ksp_strs, kspace, pat_strs, pattern);


	if (NULL == traj_file) {

		// print some statistics

		long T = md_calc_size(DIMS, pat_dims);
		long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);

		debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples);

		ifftmod(DIMS, ksp_dims, FFT_FLAGS, kspace, kspace);
		fftmod(DIMS, map_dims, FFT_FLAGS, maps, maps);
	}

	// apply fov mask to sensitivities

	if (-1. != restrict_fov) {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		apply_mask(DIMS, map_dims, maps, restrict_dims);
	}


	// initialize forward_op

	const struct linop_s* forward_op = NULL;

	if (NULL == traj_file) {

		forward_op = sense_init(shared_img_flags & ~motion_flags, max_dims, map_flags, maps);

		// apply temporal basis

		if (NULL != basis_file) {

			const struct linop_s* basis_op = linop_fmac_create(DIMS, bmx_dims, COEFF_FLAG, TE_FLAG, ~(COEFF_FLAG | TE_FLAG), basis);
			forward_op = linop_chain_FF(forward_op, basis_op);
		}

		auto cod = linop_codomain(forward_op);
		const struct linop_s* sample_op = linop_sampling_create(cod->dims, pat_dims, pattern);
		forward_op = linop_chain_FF(forward_op, sample_op);

	} else {

		const struct linop_s* nufft_op = NULL;

		if ((NULL != psf_ifile) && (NULL == psf_ofile))
			nuconf.nopsf = true;

		const complex float* traj_tmp = traj;

		//for computation of psf on GPU
#ifdef USE_CUDA
		if (gpu_gridding) {

			assert(conf.gpu);

			traj_tmp = md_gpu_move(DIMS, traj_dims, traj, CFL_SIZE);
		}
#endif

		forward_op = sense_nc_init(max_dims, map_dims, maps, ksp_dims,
				traj_dims, traj_tmp, nuconf,
				pat_dims, pattern,
				basis_dims, basis, &nufft_op, shared_img_flags & ~motion_flags);

#ifdef USE_CUDA
		if (gpu_gridding)
			md_free(traj_tmp);
#endif

		if (NULL != psf_ofile) {

			int D = nufft_get_psf_dims(nufft_op, 0, NULL);

			long psf_dims[D];

			nufft_get_psf_dims(nufft_op, D, psf_dims);

			complex float* psf_out = create_cfl_wrap(psf_ofile, D, psf_dims, hint);

			nufft_get_psf(nufft_op, D, psf_dims, psf_out);

			unmap_cfl(D, psf_dims, psf_out);

		}

		if (NULL != psf_ifile) {

			long psf_dims[DIMS + 1];

			complex float* psf_in = load_cfl_wrap(psf_ifile, DIMS + 1, psf_dims, hint);

			nufft_update_psf(nufft_op, DIMS + 1, psf_dims, psf_in);

			unmap_cfl(DIMS + 1, psf_dims, psf_in);
		}

		linop_free(nufft_op);
	}

	if (NULL != motion) {

		long img_motion_dims[DIMS];
		md_copy_dims(DIMS, img_motion_dims, img_dims);
		md_max_dims(DIMS, ~MOTION_FLAG, img_motion_dims, img_motion_dims, motion_dims);

		const struct linop_s* motion_op = linop_interpolate_displacement_create(MOTION_DIM, (1 == img_dims[2]) ? 3 : 7, 1, DIMS, img_motion_dims, motion_dims, motion, img_dims);

		forward_op = linop_chain_FF(motion_op, forward_op);
	}

	if (conf.rvc)
		forward_op = linop_chain_FF(linop_realval_create(DIMS, img_dims), forward_op);

#ifdef USE_CUDA
	if (conf.gpu && (gpu_gridding || NULL == traj)) {

		auto tmp = linop_gpu_wrapper(forward_op);
		linop_free(forward_op);
		forward_op = tmp;
	}
#endif

	// apply scaling

	if (0. == scaling) {

		if (NULL == traj_file) {

			scaling = estimate_scaling(ksp_dims, NULL, kspace);

		} else {

			complex float* adj = md_alloc_sameplace(DIMS, img_dims, CFL_SIZE, kspace);

			linop_adjoint(forward_op, DIMS, img_dims, adj, DIMS, ksp_dims, kspace);

			scaling = estimate_scaling_norm(1., md_calc_size(DIMS, img_dims), adj, false);

			md_free(adj);
		}
	}

	if (0. == scaling) {

		debug_printf(DP_WARN, "Estimated scale is zero. Set to one.");

		scaling = 1.;

	} else {

		debug_printf(DP_DEBUG1, "Inverse scaling of the data: %f\n", scaling);
		md_zsmul(DIMS, ksp_dims, kspace, kspace, 1. / scaling);

		if (conf.bpsense) {

			bpsense_eps /= scaling;
			debug_printf(DP_DEBUG1, "scaling basis pursuit eps: %.3e\n", bpsense_eps);
		}
		pridu.sigma_tau_ratio = scaling;
	}

	if (ropts.teasl) {

		const struct linop_s* hadamard_op = linop_hadamard_create(DIMS, img_dims, ITER_DIM);
		forward_op = linop_chain_FF(hadamard_op, forward_op);
	}

	complex float* image = create_cfl_wrap(out_file, DIMS, img_dims, hint);
	md_clear(DIMS, img_dims, image, CFL_SIZE);


	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (NULL != image_truth_file) {

		debug_printf(DP_INFO, "Compare to truth\n");

		image_truth = load_cfl_wrap(image_truth_file, DIMS, img_truth_dims, hint);
		//md_zsmul(DIMS, img_dims, image_truth, image_truth, 1. / scaling);

#ifdef USE_CUDA
		if (conf.gpu) {

			complex float* gpu_image_truth = md_gpu_move(DIMS, img_dims, image_truth, CFL_SIZE);

			unmap_cfl(DIMS, img_dims, image_truth);

			image_truth = gpu_image_truth;
		}
#endif
		xfree(image_truth_file);
	}

	long img_start_dims[DIMS];
	complex float* image_start = NULL;

	if (NULL != image_start_file) {

		debug_printf(DP_DEBUG1, "Warm start: %s\n", image_start_file);

		image_start = load_cfl_wrap(image_start_file, DIMS, img_start_dims, hint);

		assert(md_check_compat(DIMS, 0u, img_start_dims, img_dims));

		// if rescaling at the end, assume the input has also been rescaled
		if (scale_im && (scaling != 0.))
			md_zsmul(DIMS, img_dims, image_start, image_start, 1. / scaling);
	}

	double maxeigen = 1.;

	if (eigen && (ALGO_PRIDU != algo)) {

		// Maxeigen in PRIDU must include regularizations
		maxeigen = estimate_maxeigenval_sameplace(forward_op->normal, 30, kspace);

		debug_printf(DP_INFO, "Maximum eigenvalue: %.2e\n", maxeigen);
	}


	// initialize prox functions

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };
	const long (*sdims[NUM_REGS])[DIMS + 1] = { NULL };


	opt_reg_configure(DIMS, img_dims, &ropts, thresh_ops, trafos, sdims, llr_blk, shift_mode, wtype_str, conf.gpu, ITER_DIM);

	if (conf.bpsense)
		opt_bpursuit_configure(&ropts, thresh_ops, trafos, forward_op, kspace, bpsense_eps);

	if (conf.precond)
		opt_precond_configure(&ropts, thresh_ops, trafos, forward_op, DIMS, ksp_dims, kspace, pat_dims, conf.precond ? pattern : NULL);

	int nr_penalties = ropts.r + ropts.sr;

	debug_printf(DP_INFO, "Regularization terms: %d, Supporting variables: %ld\n", nr_penalties, ropts.svars);

	// choose algorithm

	if (ALGO_DEFAULT == algo)
		algo = italgo_choose(ropts.r, ropts.regs);

	if (conf.bpsense || conf.precond)
		assert((ALGO_ADMM == algo) || (ALGO_PRIDU == algo));


	// choose step size

	if ((ALGO_IST == algo) || (ALGO_FISTA == algo) || (ALGO_PRIDU == algo)) {

		// For non-Cartesian trajectories, the default
		// will usually not work. TODO: The same is true
		// for sensitivities which are not normalized, but
		// we do not detect this case.

		if ((NULL != traj_file) && (-1. == step) && !eigen)
			debug_printf(DP_WARN, "No step size specified.\n");

		if (-1. == step)
			step = 0.95;
	}

	if ((ALGO_CG == algo) || (ALGO_ADMM == algo))
		if (-1. != step)
			debug_printf(DP_INFO, "Stepsize ignored.\n");

	step /= maxeigen;


	// initialize algorithm
	pridu.maxeigen_iter = eigen ? 30 : 0;

	struct iter it = italgo_config(algo, nr_penalties, ropts.regs, maxiter, step, hogwild, admm, fista, pridu, NULL != image_truth);

	if (ALGO_CG == algo)
		nr_penalties = 0;

	bool trafos_cond = (   (ALGO_PRIDU == algo)
			    || (ALGO_ADMM == algo)
			    || (   (ALGO_NIHT == algo)
				&& (ropts.regs[0].xform == NIHTWAV)));

	struct iter_monitor_s* monitor = NULL;

	if (NULL != image_truth)
		monitor = iter_monitor_create(2 * md_calc_size(DIMS, img_dims), (const float*)image_truth, NULL, NULL);

	const void* vptr_ref = NULL;

	if (0 < ropts.svars) {

		assert(NULL == image_truth);
		assert(!conf.rvc);

		const struct linop_s* extract = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(DIMS, img_dims)), MD_DIMS(md_calc_size(DIMS, img_dims) + ropts.svars));
		extract = linop_reshape_out_F(extract, DIMS, img_dims);
		forward_op = linop_chain_FF(extract, forward_op);

		if (is_vptr(image)) {

			void* vptr_ref_array[NUM_REGS] = { NULL };
			vptr_ref_array[0] = vptr_alloc_same(image);
			int svars = 1;
			for (int i = 0; i < NUM_REGS; i++)
				if (NULL != sdims[i])
					vptr_ref_array[svars++] = vptr_alloc_sameplace(DIMS + 1, (*sdims[i]), CFL_SIZE, image);

			vptr_ref = vptr_wrap_range(svars, vptr_ref_array, true);

			const struct linop_s* tmp = linop_vptr_set_dims_wrapper((struct linop_s*)forward_op, NULL, vptr_ref, vptr_get_hint(image));
			linop_free(forward_op);
			forward_op = tmp;
		}
	}

	const struct operator_p_s* po = sense_recon_create(&conf, forward_op,
				pat_dims,
				it.italgo, it.iconf, image_start, nr_penalties, thresh_ops,
				trafos_cond ? trafos : NULL, NULL, monitor);

	const struct operator_s* op = operator_p_bind(po, 1.);
	operator_p_free(po);

	if (0 < ropts.svars) {

		const struct linop_s* extract = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(DIMS, img_dims)), MD_DIMS(md_calc_size(DIMS, img_dims) + ropts.svars));
		extract = linop_reshape_out_F(extract, DIMS, img_dims);

		if (is_vptr(image)) {

			const struct linop_s* tmp = linop_vptr_set_dims_wrapper((struct linop_s*)extract, NULL, vptr_ref, vptr_get_hint(image));
			linop_free(extract);
			extract = tmp;

			auto op2 = operator_vptr_set_dims_wrapper(op, 2, (const void*[2]){ vptr_ref, NULL }, vptr_get_hint(image));
			operator_free(op);
			op = op2;
		}

		auto op2 = operator_chain(op, extract->forward);

		operator_free(op);
		op = op2;

		linop_free(extract);
	}

	for (int i = 0; i < NUM_REGS; i++)
		if (NULL != sdims[i])
			xfree(sdims[i]);


	auto iov = operator_domain(op);
	operator_apply(op, DIMS, img_dims, image, (conf.bpsense || conf.precond) ? iov->N : DIMS, (conf.bpsense || conf.precond) ? iov->dims : ksp_dims, (conf.bpsense || conf.precond) ? NULL : kspace);

	md_free(vptr_ref);

	operator_free(op);

	opt_reg_free(&ropts, thresh_ops, trafos);

	italgo_config_free(it);

	if (scale_im)
		md_zsmul(DIMS, img_dims, image, image, scaling);

	// clean up

	if (NULL != pat_file)
		unmap_cfl(DIMS, pat_dims, pattern);
	else
		md_free(pattern);

	unmap_cfl(DIMS, img_dims, image);

	if (NULL != image_truth) {

#ifdef USE_CUDA
		if (conf.gpu)
			md_free(image_truth);
		else
#endif
			unmap_cfl(DIMS, img_dims, image_truth);
	}

	if (image_start)
		unmap_cfl(DIMS, img_dims, image_start);

	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, ksp_dims, kspace);

	if (NULL != traj)
		unmap_cfl(DIMS, traj_dims, traj);

	if (NULL != basis)
		unmap_cfl(DIMS, basis_dims, basis);

	vptr_hint_free(hint);

	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

	return 0;
}

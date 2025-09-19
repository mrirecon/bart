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

#include "iter/misc.h"
#include "iter/monitor.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"
#include "linops/realval.h"

#include "noncart/nufft.h"

#include "num/rand.h"
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
			const struct linop_s** fft_opp, unsigned long shared_img_dims, unsigned long lowmem_stack)
{
	lowmem_stack &= md_nontriv_dims(DIMS, max_dims);

	if (0UL != (lowmem_stack & (conf.flags | conf.cfft)))
		error("Lowmem-stacking not possible along FFT_FLAGS.\n");

	if ((NULL != basis) && (0UL != (lowmem_stack & (TE_FLAG | COEFF_FLAG))))
		error("Lowmem-stacking not possible along basis dimensions.\n");

	for (int i = DIMS - 1; i > MAPS_DIM; i--) {

		if (!MD_IS_SET(lowmem_stack, i))
			continue;

		long n_map_dims[DIMS];
		long n_max_dims[DIMS];
		long n_traj_dims[DIMS];
		long n_ksp_dims[DIMS];
		long n_wgs_dims[DIMS];
		long n_basis_dims[DIMS];

		md_select_dims(DIMS, ~MD_BIT(i), n_map_dims, map_dims);
		md_select_dims(DIMS, ~MD_BIT(i), n_max_dims, max_dims);
		md_select_dims(DIMS, ~MD_BIT(i), n_traj_dims, traj_dims);
		md_select_dims(DIMS, ~MD_BIT(i), n_ksp_dims, ksp_dims);

		if (NULL != weights)
			md_select_dims(DIMS, ~MD_BIT(i), n_wgs_dims, wgs_dims);

		if (NULL != basis)
			md_select_dims(DIMS, ~MD_BIT(i), n_basis_dims, basis_dims);

		if (DIMS != md_calc_blockdim(DIMS, n_map_dims, MD_STRIDES(DIMS, map_dims, CFL_SIZE), CFL_SIZE))
			error("Sensitivity maps not continuous for stacking along dim %d.\n");

		if (DIMS != md_calc_blockdim(DIMS, n_traj_dims, MD_STRIDES(DIMS, traj_dims, CFL_SIZE), CFL_SIZE))
			error("Trajectory not continuous for stacking along dim %d.\n");

		if ((NULL != weights) && (DIMS != md_calc_blockdim(DIMS, n_wgs_dims, MD_STRIDES(DIMS, wgs_dims, CFL_SIZE), CFL_SIZE)))
			error("Weights not continuous for stacking along dim %d.\n");

		if ((NULL != basis) && (DIMS != md_calc_blockdim(DIMS, n_basis_dims, MD_STRIDES(DIMS, basis_dims, CFL_SIZE), CFL_SIZE)))
			error("Basis not continuous for stacking along dim %d.\n");

		long offset_basis = (NULL != basis) && (1 != basis_dims[i]) ? md_calc_size(i, basis_dims) : 0;
		long offset_weights = (NULL != weights) && (1 != wgs_dims[i]) ? md_calc_size(i, wgs_dims) : 0;
		long offset_traj = (1 != traj_dims[i]) ? md_calc_size(i, traj_dims) : 0;
		long offset_sens = (1 != map_dims[i]) ? md_calc_size(i, map_dims) : 0;

		if (conf.nopsf)
			error("Lowmem stacking not compatible with precomputed psf!\n");

		debug_printf(DP_DEBUG1, "Lowmem-stacking along dim %d\n!", i);

		const struct linop_s* lop = sense_nc_init(n_max_dims, n_map_dims, maps, n_ksp_dims, n_traj_dims, traj, conf, n_wgs_dims, weights, n_basis_dims, basis, NULL, shared_img_dims, lowmem_stack);

		for (int j = 1; j < max_dims[i]; j++) {

			auto tmp = sense_nc_init(n_max_dims, n_map_dims, maps + j * offset_sens, n_ksp_dims, n_traj_dims,
									traj + j * offset_traj, conf, n_wgs_dims, weights + j * offset_weights,
									n_basis_dims, basis + j * offset_basis, NULL, shared_img_dims, lowmem_stack);
			if (MD_IS_SET(shared_img_dims, i))
				lop = linop_stack_cod_F(2, MAKE_ARRAY(lop, tmp), i);
			else
				lop = linop_stack_FF(i, i, lop, tmp);
		}

		return lop;
	}

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

	if (MD_IS_SET(lowmem_stack, COIL_DIM)) {

		long map_dims_slc[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, map_dims_slc, map_dims);

		if (!md_check_equal_dims(DIMS, MD_STRIDES(DIMS, map_dims_slc, CFL_SIZE), map_strs, ~COIL_FLAG)) {

			lowmem_stack = 0;
			debug_printf(DP_WARN, "Lowmem-stacking currently only supported for continuous map dims, set stacking flag to %lu!\n", lowmem_stack);

		} else {

			ksp_dims2[COIL_DIM] = 1;
			coilim_dims[COIL_DIM] = 1;
		}
	}

	const struct linop_s* nufft_op = nufft_create2(DIMS, ksp_dims2, coilim_dims,
						traj_dims, traj,
						(weights ? wgs_dims : NULL), weights,
						(basis ? basis_dims : NULL), basis, conf);

	const struct linop_s* lop;

	if (MD_IS_SET(lowmem_stack, COIL_DIM)) {

		long map_dims_slc[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, map_dims_slc, map_dims);

		const struct linop_s* lops[map_dims[COIL_DIM]];

		for (int i = 0; i < map_dims[COIL_DIM]; i++) {

			const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims_slc, img_dims, maps + i *  map_strs[COIL_DIM] / (long)CFL_SIZE);
			lops[i] = linop_chain(maps_op, nufft_op);
			linop_free(maps_op);
		}

		lop = linop_stack_cod_F(map_dims[COIL_DIM], lops, COIL_DIM);

	} else {

		const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims, img_dims, maps);
		lop = linop_chain(maps_op, nufft_op);
		linop_free(maps_op);
	}

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
	unsigned long lowmem_flags = 0UL;

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
		OPTL_SET('U', "lowmem", &nufft_conf_options.lowmem, "Use low-mem mode of the nuFFT"),
		OPTL_ULONG(0, "lowmem-stack", &lowmem_flags, "flags", "(Stack SENSE model along selected dimscurrently only supports COIL_DIM and noncart)"),
		OPTL_CLEAR(0, "no-toeplitz", &nufft_conf_options.toeplitz, "(Turn off Toeplitz mode of nuFFT)"),
		OPTL_OUTFILE(0, "psf_export", &psf_ofile, "file", "Export PSF to file"),
		OPTL_INFILE(0, "psf_import", &psf_ifile, "file", "Import PSF from file"),
		OPTL_STRING(0, "wavelet", &wtype_str, "name", "wavelet type (haar,dau2,cdf44)"),
		OPTL_ULONG(0, "mpi", &mpi_flags, "flags", "distribute over this dimensions with use of MPI"),
		OPTL_FLVEC3(0, "fista_pqr", &fista.params, "p:q:r", "parameters for FISTA acceleration"),
		OPTL_SET(0, "fista_last", &fista.last, "end iteration with call to data consistency"),
		OPTL_INFILE(0, "motion-field", &motion_file, "file", "motion field"),
		OPTL_SUBOPT(0, "nufft-conf", "...", "configure nufft", N_nufft_conf_opts, nufft_conf_opts),
	};


	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	bool use_mpi = (0 != mpi_flags);

	if (use_mpi) {

#ifndef USE_MPI
		error("Compiled without MPI support\n");
#endif
		if (1 == mpi_get_num_procs())
			error("MPI requested but not initialized using bart wrapper!\n");

		if (cfl_loop_desc_active())
			error("Simultaneous use of BART generic looping interface and --mpi not supported!\n");
	}

	if (0 != loop_flags)
		error("Looping only supported via BART generic looping interface!\n");

	if (0 <= bpsense_eps)
		conf.bpsense = true;

	admm.dynamic_tau = admm.relative_norm;

	struct nufft_conf_s nuconf = nufft_conf_options;

	if (conf.bpsense)
		nuconf.toeplitz = false;

	if (0 != lowmem_flags) {

		nuconf.lowmem = true;
		nuconf.precomp_fftmod = false;
		nuconf.precomp_roll = !nuconf.toeplitz;
		nuconf.precomp_linphase = false;
	}


	long max_dims[DIMS];
	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long coilim_dims[DIMS];
	long ksp_dims[DIMS];
	long traj_dims[DIMS];

	memset(traj_dims, 0, sizeof traj_dims);	// GCC ANALYZER


	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

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

	complex float* maps = load_cfl(sens_file, DIMS, map_dims);

	unsigned long map_flags = md_nontriv_dims(DIMS, map_dims);

	map_flags |= FFT_FLAGS | SENS_FLAGS;



	long basis_dims[DIMS] = { }; // analyzer false positive
	complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, basis_dims);

		assert(!md_check_dimensions(DIMS, basis_dims, COEFF_FLAG | TE_FLAG));
	}

	long motion_dims[DIMS] = { };
	complex float* motion = NULL;
	unsigned long motion_flags = 0;

	if (NULL != motion_file) {

		motion = load_cfl(motion_file, DIMS, motion_dims);
		assert(1 < motion_dims[MOTION_DIM]);
		motion_flags = md_nontriv_dims(DIMS, motion_dims) & ~MOTION_FLAG;
	}


	complex float* traj = NULL;

	if (NULL != traj_file)
		traj = load_cfl(traj_file, DIMS, traj_dims);

	complex float* kspace_p = kspace;
	complex float* maps_p = maps;
	complex float* traj_p = traj;

	struct vptr_hint_s* hint = NULL;

	if (use_mpi) {

		hint = hint_mpi_create(mpi_flags, DIMS, ksp_dims);

		kspace_p = vptr_wrap(DIMS, ksp_dims, CFL_SIZE, kspace, hint, false, false);

		if (NULL != traj)
			traj_p = vptr_wrap(DIMS, traj_dims, CFL_SIZE, traj, hint, false, false);

		maps_p = vptr_wrap(DIMS, map_dims, CFL_SIZE, maps, hint, false, false);
	}


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

	num_rand_init(0ULL);

	num_init_gpu_support();
	conf.gpu = bart_use_gpu;

	// print options

	if (use_mpi)
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

		pattern = load_cfl(pat_file, DIMS, pat_dims);

		assert(md_check_compat(DIMS, COIL_FLAG, ksp_dims, pat_dims));

	} else {

		md_select_dims(DIMS, ~COIL_FLAG, pat_dims, ksp_dims);
		pattern = md_alloc_sameplace(DIMS, pat_dims, CFL_SIZE, kspace_p);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace_p);
	}


	long ksp_strs[DIMS];
	long pat_strs[DIMS];

	md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);
	md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

	md_zmul2(DIMS, ksp_dims, ksp_strs, kspace_p, ksp_strs, kspace_p, pat_strs, pattern);


	if (NULL == traj_file) {

		// print some statistics

		long T = md_calc_size(DIMS, pat_dims);
		long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);

		debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples);

		ifftmod(DIMS, ksp_dims, FFT_FLAGS, kspace_p, kspace_p);
		fftmod(DIMS, map_dims, FFT_FLAGS, maps_p, maps_p);
	}

	// apply fov mask to sensitivities

	if (-1. != restrict_fov) {

		float restrict_dims[DIMS] = { [0 ... DIMS - 1] = 1. };
		restrict_dims[0] = restrict_fov;
		restrict_dims[1] = restrict_fov;
		restrict_dims[2] = restrict_fov;

		apply_mask(DIMS, map_dims, maps_p, restrict_dims);
	}


	// initialize forward_op

	const struct linop_s* forward_op = NULL;

	if (NULL == traj_file) {

		forward_op = sense_init(shared_img_flags & ~motion_flags, max_dims, map_flags, maps_p);

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

		const complex float* traj_tmp = traj_p;

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
				basis_dims, basis, &nufft_op, shared_img_flags & ~motion_flags, lowmem_flags);

#ifdef USE_CUDA
		if (gpu_gridding)
			md_free(traj_tmp);
#endif

		if (NULL != psf_ofile) {

			int D = nufft_get_psf_dims(nufft_op, 0, NULL);

			long psf_dims[D];

			nufft_get_psf_dims(nufft_op, D, psf_dims);

			complex float* psf_out = create_cfl(psf_ofile, D, psf_dims);

			nufft_get_psf(nufft_op, D, psf_dims, psf_out);

			unmap_cfl(D, psf_dims, psf_out);

		}

		if (NULL != psf_ifile) {

			long psf_dims[DIMS + 1];

			complex float* psf_in = load_cfl(psf_ifile, DIMS + 1, psf_dims);

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

	if (NULL != hint) {

		auto tmp = linop_vptr_wrapper(hint, forward_op);
		linop_free(forward_op);
		forward_op = tmp;

		vptr_hint_free(hint);
	}

	// apply scaling

	if (0. == scaling) {

		if (NULL == traj_file) {

			scaling = estimate_scaling(ksp_dims, NULL, kspace_p);

		} else {

			complex float* adj = md_alloc(DIMS, img_dims, CFL_SIZE);

			linop_adjoint(forward_op, DIMS, img_dims, adj, DIMS, ksp_dims, kspace_p);

			scaling = estimate_scaling_norm(1., md_calc_size(DIMS, img_dims), adj, false);

			md_free(adj);
		}
	}

	if (0. == scaling) {

		debug_printf(DP_WARN, "Estimated scale is zero. Set to one.");

		scaling = 1.;

	} else {

		debug_printf(DP_DEBUG1, "Inverse scaling of the data: %f\n", scaling);

		md_zsmul(DIMS, ksp_dims, kspace_p, kspace_p, 1. / scaling);

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

	complex float* image = create_cfl(out_file, DIMS, img_dims);

	md_clear(DIMS, img_dims, image, CFL_SIZE);


	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (NULL != image_truth_file) {

		debug_printf(DP_INFO, "Compare to truth\n");

		image_truth = load_cfl(image_truth_file, DIMS, img_truth_dims);
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

		image_start = load_cfl(image_start_file, DIMS, img_start_dims);

		assert(md_check_compat(DIMS, 0u, img_start_dims, img_dims));

		// if rescaling at the end, assume the input has also been rescaled
		if (scale_im && (scaling != 0.))
			md_zsmul(DIMS, img_dims, image_start, image_start, 1. / scaling);
	}

	double maxeigen = 1.;

	if (eigen && (ALGO_PRIDU != algo)) {

		// Maxeigen in PRIDU must include regularizations
		maxeigen = estimate_maxeigenval(forward_op->normal);

		debug_printf(DP_INFO, "Maximum eigenvalue: %.2e\n", maxeigen);
	}


	// initialize prox functions

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };


	opt_reg_configure(DIMS, img_dims, &ropts, thresh_ops, trafos, NULL, llr_blk, shift_mode, wtype_str, conf.gpu, ITER_DIM);

	if (conf.bpsense)
		opt_bpursuit_configure(&ropts, thresh_ops, trafos, forward_op, kspace_p, bpsense_eps);

	if (conf.precond)
		opt_precond_configure(&ropts, thresh_ops, trafos, forward_op, DIMS, ksp_dims, kspace_p, pat_dims, conf.precond ? pattern : NULL);

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

	if (0 < ropts.svars) {

		assert(NULL == image_truth);
		assert(!conf.rvc);

		const struct linop_s* extract = linop_extract_create(1, MD_DIMS(0), MD_DIMS(md_calc_size(DIMS, img_dims)), MD_DIMS(md_calc_size(DIMS, img_dims) + ropts.svars));
		extract = linop_reshape_out_F(extract, DIMS, img_dims);
		forward_op = linop_chain_FF(extract, forward_op);
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

		auto op2 = operator_chain(op, extract->forward);

		operator_free(op);
		op = op2;

		linop_free(extract);
	}


	auto iov = operator_domain(op);
	operator_apply(op, DIMS, img_dims, image, (conf.bpsense || conf.precond) ? iov->N : DIMS, (conf.bpsense || conf.precond) ? iov->dims : ksp_dims, (conf.bpsense || conf.precond) ? NULL : kspace_p);

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
	unmap_cfl(DIMS, img_dims, image_start);

#ifdef USE_CUDA
	if (conf.gpu)
		md_free(image_truth);
	else
#endif
		unmap_cfl(DIMS, img_dims, image_truth);

	if (kspace_p != kspace)
		md_free(kspace_p);

	if (maps_p != maps)
		md_free(maps_p);

	if (traj_p != traj)
		md_free(traj_p);

	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, traj_dims, traj);

	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

	return 0;
}


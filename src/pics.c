/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2015-2022. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2022 Martin Uecker
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

#include "iter/misc.h"
#include "iter/monitor.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "noncart/nufft.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "num/iovec.h"
#include "num/ops.h"

static const char help_str[] = "Parallel-imaging compressed-sensing reconstruction.\n";
                 

static const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps, const long ksp_dims[DIMS],
			const long traj_dims[DIMS], const complex float* traj, struct nufft_conf_s conf,
			const long wgs_dims[DIMS], const complex float* weights,
			const long basis_dims[DIMS], const complex float* basis,
			const struct linop_s** fft_opp, unsigned long shared_img_dims, unsigned long lowmem_stack)
{
	lowmem_stack &= md_nontriv_dims(DIMS, max_dims);

	if (0 != (lowmem_stack & (conf.flags | conf.cfft))) {

		lowmem_stack = lowmem_stack & ~(conf.flags | conf.cfft);
		debug_printf(DP_WARN, "Lowmem-stacking not possible along FFT_FLAGS, set stacking flag to %lu!\n", lowmem_stack);
	}

	if ((NULL != basis) && (0 != (lowmem_stack & (TE_FLAG | COEFF_FLAG)))) {

		lowmem_stack = lowmem_stack & ~(TE_FLAG | COEFF_FLAG);
		debug_printf(DP_WARN, "Lowmem-stacking not possible along basis dimensions, set stacking flag to %lu!\n", lowmem_stack);
	}

	for (int i = DIMS - 1; i > MAPS_DIM; i--) {

		if (MD_IS_SET(lowmem_stack, i)) {

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

			if (DIMS != md_calc_blockdim(DIMS, n_map_dims, MD_STRIDES(DIMS, map_dims, CFL_SIZE), CFL_SIZE)) {

				lowmem_stack &= ~MD_BIT(i);
				debug_printf(DP_WARN, "Sensitivity maps not continuous for stacking along dim %d, set stacking flag to %lu!\n", lowmem_stack);
				continue;
			}

			if (DIMS != md_calc_blockdim(DIMS, n_traj_dims, MD_STRIDES(DIMS, traj_dims, CFL_SIZE), CFL_SIZE)) {

				lowmem_stack &= ~MD_BIT(i);
				debug_printf(DP_WARN, "Trajectory not continuous for stacking along dim %d, set stacking flag to %lu!\n", lowmem_stack);
				continue;
			}

			if ((NULL != weights) && (DIMS != md_calc_blockdim(DIMS, n_wgs_dims, MD_STRIDES(DIMS, wgs_dims, CFL_SIZE), CFL_SIZE))) {

				lowmem_stack &= ~MD_BIT(i);
				debug_printf(DP_WARN, "Weights not continuous for stacking along dim %d, set stacking flag to %lu!\n", lowmem_stack);
				continue;
			}

			if ((NULL != basis) && (DIMS != md_calc_blockdim(DIMS, n_basis_dims, MD_STRIDES(DIMS, basis_dims, CFL_SIZE), CFL_SIZE))) {

				lowmem_stack &= ~MD_BIT(i);
				debug_printf(DP_WARN, "Basis not continuous for stacking along dim %d, set stacking flag to %lu!\n", lowmem_stack);
				continue;
			}

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
					lop = linop_stack_cod_F(2, (struct linop_s*[2]){ (struct linop_s*)lop, (struct linop_s*)tmp }, i);
				else
					lop = linop_stack_FF(i, i, lop, tmp);
			}

			return lop;
		}
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

		struct linop_s* lops[map_dims[COIL_DIM]];

		for (int i = 0; i < map_dims[COIL_DIM]; i++) {

			const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims_slc, img_dims, maps + i *  map_strs[COIL_DIM] / CFL_SIZE);
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

	unsigned int shift_mode = 0;
	bool randshift = true;
	bool overlapping_blocks = false;
	unsigned int maxiter = 30;
	float step = -1.;

	// Start time count

	double start_time = timestamp();

	// Read input options
	struct nufft_conf_s nuconf = nufft_conf_defaults;
	nuconf.toeplitz = true;
	nuconf.lowmem = false;

	float restrict_fov = -1.;
	const char* pat_file = NULL;
	const char* traj_file = NULL;
	const char* psf_ifile = NULL;
	const char* psf_ofile = NULL;
	bool scale_im = false;
	bool eigen = false;
	float scaling = 0.;

        // Simultaneous Multi-Slice
        bool sms = false;

	unsigned int llr_blk = 8;
	const char* wtype_str = "dau2";

	const char* image_truth_file = NULL;
	bool im_truth = false;

	const char* image_start_file = NULL;
	bool warm_start = false;

	const char* basis_file = NULL;

	struct admm_conf admm = { false, false, false, iter_admm_defaults.rho, iter_admm_defaults.maxitercg };

	enum algo_t algo = ALGO_DEFAULT;

	bool hogwild = false;
	bool fast = false;

	bool gpu_gridding = false;
	unsigned int requested_gpus = 0u;

	struct opt_reg_s ropts;
	opt_reg_init(&ropts);

	unsigned long loop_flags = 0UL;
	unsigned long shared_img_flags = 0UL;
	unsigned long lowmem_flags = 0UL;

	const struct opt_s opts[] = {

		{ 'l', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "\b1/-l2", "  toggle l1-wavelet or l2 regularization." },
		OPT_FLOAT('r', &ropts.lambda, "lambda", "regularization parameter"),
		{ 'R', NULL, true, OPT_SPECIAL, opt_reg, &ropts, "<T>:A:B:C", "generalized regularization options (-Rh for help)" },
		OPT_SET('c', &conf.rvc, "real-value constraint"),
		OPT_FLOAT('s', &step, "step", "iteration stepsize"),
		OPT_UINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_INFILE('t', &traj_file, "file", "k-space trajectory"),
		OPT_CLEAR('n', &randshift, "disable random wavelet cycle spinning"),
		OPT_SET('N', &overlapping_blocks, "do fully overlapping LLR blocks"),
		OPT_SET('g', &conf.gpu, "use GPU"),
		OPTL_SET(0, "gpu-gridding", &gpu_gridding, "use GPU for gridding"),
		OPT_UINT('G', &requested_gpus, "bitmask", "bitmask of requested GPU devices"),
		OPT_INFILE('p', &pat_file, "file", "pattern or weights"),
		OPTL_SET(0, "precond", &(conf.precond), "interprete weights as preconditioner"),
		OPT_UINT('b', &llr_blk, "blk", "Lowrank block size"),
		OPT_SET('e', &eigen, "Scale stepsize based on max. eigenvalue"),
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_SET('D', &admm.dynamic_rho, "(ADMM dynamic step size)"),
		OPT_SET('F', &fast, "(fast)"),
		OPT_SET('J', &admm.relative_norm, "(ADMM residual balancing)"),
		OPT_INFILE('T', &image_truth_file, "file", "(truth file)"),
		OPT_INFILE('W', &image_start_file, "<img>", "Warm start with <img>"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_INT('O', &conf.rwiter, "rwiter", "(reweighting)"),
		OPT_FLOAT('o', &conf.gamma, "gamma", "(reweighting)"),
		OPT_FLOAT('u', &admm.rho, "rho", "ADMM rho"),
		OPT_UINT('C', &admm.maxitercg, "iter", "ADMM max. CG iterations"),
		OPT_FLOAT('q', &conf.cclambda, "cclambda", "(cclambda)"),
		OPT_FLOAT('f', &restrict_fov, "rfov", "restrict FOV"),
		OPTL_SELECT('I', "ist", enum algo_t, &algo, ALGO_IST, "select IST"),
		OPTL_SELECT(0, "fista", enum algo_t, &algo, ALGO_FISTA, "select FISTA"),
		OPTL_SELECT('m', "admm", enum algo_t, &algo, ALGO_ADMM, "select ADMM"),
		OPTL_SELECT('a', "pridu", enum algo_t, &algo, ALGO_PRIDU, "select Primal Dual"),
		OPT_FLOAT('w', &scaling, "", "inverse scaling of the data"),
		OPT_SET('S', &scale_im, "re-scale the image after reconstruction"),
		OPT_ULONG('L', &loop_flags, "flags", "batch-mode"),
		OPTL_ULONG(0, "shared-img-dims", &shared_img_flags, "flags", "deselect image dims with flags"),
		OPT_SET('K', &nuconf.pcycle, "randshift for NUFFT"),
		OPT_INFILE('B', &basis_file, "file", "temporal (or other) basis"),
		OPT_FLOAT('P', &bpsense_eps, "eps", "Basis Pursuit formulation, || y- Ax ||_2 <= eps"),
		OPT_SET('M', &sms, "Simultaneous Multi-Slice reconstruction"),
		OPTL_SET('U', "lowmem", &nuconf.lowmem, "Use low-mem mode of the nuFFT"),
		OPTL_ULONG(0, "lowmem-stack", &lowmem_flags, "flags", "(Stack SENSE model along selected dims (currently only supports COIL_DIM and noncart)))"),
		OPTL_CLEAR(0, "no-toeplitz", &nuconf.toeplitz, "Turn off Toeplitz mode of nuFFT"),
		OPTL_OUTFILE(0, "psf_export", &psf_ofile, "file", "Export PSF to file"),
		OPTL_INFILE(0, "psf_import", &psf_ifile, "file", "Import PSF from file"),
		OPTL_STRING(0, "wavelet", &wtype_str, "name", "wavelet type (haar,dau2,cdf44)"),
	};


	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (NULL != image_truth_file)
		im_truth = true;

	if (NULL != image_start_file)
		warm_start = true;

	if (0 <= bpsense_eps)
		conf.bpsense = true;

	admm.dynamic_tau = admm.relative_norm;

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


	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(ksp_file, DIMS, ksp_dims);

        if (sms) {

		if (NULL == traj_file)
			error("SMS is only supported for non-Cartesian trajectories.\n");

		nuconf.cfft |= SLICE_FLAG;

                debug_printf(DP_INFO, "SMS reconstruction: MB = %ld\n", ksp_dims[SLICE_DIM]);
        }

	complex float* maps = load_cfl(sens_file, DIMS, map_dims);

	unsigned long map_flags = md_nontriv_dims(DIMS, map_dims);

	map_flags |= FFT_FLAGS | SENS_FLAGS;



	long basis_dims[DIMS] = { 0 }; // analyzer false positive
	complex float* basis = NULL;

	if (NULL != basis_file) {

		basis = load_cfl(basis_file, DIMS, basis_dims);

		assert(!md_check_dimensions(DIMS, basis_dims, COEFF_FLAG | TE_FLAG));
	}


	complex float* traj = NULL;

	if (NULL != traj_file)
		traj = load_cfl(traj_file, DIMS, traj_dims);


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

	if ((NULL == traj_file) && (0 != shared_img_flags))
		error("Shared image flags only supported for non-Cartesian trajectories.");

	md_select_dims(DIMS, ~COIL_FLAG & ~shared_img_flags, img_dims, max_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);

	if (!md_check_compat(DIMS, ~(MD_BIT(MAPS_DIM)|FFT_FLAGS), img_dims, map_dims))
		error("Dimensions of image and sensitivities do not match!\n");

	if ((NULL != traj_file) && (!md_check_compat(DIMS, ~0, ksp_dims, traj_dims)))
		error("Dimensions of data and trajectory do not match!\n");

	if ((NULL == traj_file) && (NULL != psf_ofile))
		error("Export of PSF for Cartesian scan requested!\n");


	assert(1 == ksp_dims[MAPS_DIM]);

	if (conf.gpu) {

		if (0u == requested_gpus)
			num_init_gpu_memopt();
		else
			num_init_multigpu_select(requested_gpus);
	} else {

		num_init();
	}

	// print options

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

	if (im_truth)
		debug_printf(DP_INFO, "Compare to truth\n");

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
		pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);
		estimate_pattern(DIMS, ksp_dims, COIL_FLAG, pattern, kspace);
	}


	if (NULL != traj_file) {

		if ((NULL == pat_file) && (NULL == basis)) {

			md_free(pattern);
			pattern = NULL;

		} else {

			long ksp_strs[DIMS];
			md_calc_strides(DIMS, ksp_strs, ksp_dims, CFL_SIZE);

			long pat_strs[DIMS];
			md_calc_strides(DIMS, pat_strs, pat_dims, CFL_SIZE);

			md_zmul2(DIMS, ksp_dims, ksp_strs, kspace, ksp_strs, kspace, pat_strs, pattern);
		}

	} else {

		// print some statistics

		long T = md_calc_size(DIMS, pat_dims);
		long samples = (long)pow(md_znorm(DIMS, pat_dims, pattern), 2.);

		debug_printf(DP_INFO, "Size: %ld Samples: %ld Acc: %.2f\n", T, samples, (float)T / (float)samples);
	}

	if (NULL == traj_file) {

		fftmod(DIMS, ksp_dims, FFT_FLAGS, kspace, kspace);
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

		forward_op = sense_init(max_dims, map_flags, maps);

		// apply temporal basis

		if (NULL != basis_file) {

			const struct linop_s* basis_op = linop_fmac_create(DIMS, bmx_dims, COEFF_FLAG, TE_FLAG, ~(COEFF_FLAG | TE_FLAG), basis);
			forward_op = linop_chain_FF(forward_op, basis_op);
		}

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
				basis_dims, basis, &nufft_op, shared_img_flags, lowmem_flags);

#ifdef USE_CUDA
		if (gpu_gridding) {

			md_free(traj_tmp);

			auto tmp = linop_gpu_wrapper((struct linop_s*)forward_op);

			linop_free(forward_op);

			forward_op = tmp;
		} 
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


	// apply scaling

	if (0. == scaling) {

		if (NULL == traj_file) {

			scaling = estimate_scaling(ksp_dims, NULL, kspace);

		} else {

			complex float* adj = md_alloc(DIMS, img_dims, CFL_SIZE);

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
	}


	complex float* image = create_cfl(out_file, DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);


	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (im_truth) {

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

	if (warm_start) {

		debug_printf(DP_DEBUG1, "Warm start: %s\n", image_start_file);

		image_start = load_cfl(image_start_file, DIMS, img_start_dims);

		assert(md_check_compat(DIMS, 0u, img_start_dims, img_dims));

		// if rescaling at the end, assume the input has also been rescaled
		if (scale_im && (scaling != 0.))
			md_zsmul(DIMS, img_dims, image_start, image_start, 1. / scaling);
	}



	assert((0u == loop_flags) || (NULL == image_truth));
	assert((0u == loop_flags) || (NULL == image_start));
	assert((0u == loop_flags) || (NULL == traj_file));
	assert(!(loop_flags & COIL_FLAG));

	const complex float* image_start1 = image_start;

	long loop_dims[DIMS];
	md_select_dims(DIMS,  loop_flags, loop_dims, max_dims);

	long img1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, img1_dims, img_dims);

	long ksp1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, ksp1_dims, ksp_dims);

	long max1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, max1_dims, max_dims);

	long pat1_dims[DIMS];
	md_select_dims(DIMS, ~loop_flags, pat1_dims, pat_dims);

	complex float* pattern1 = NULL;

	if (NULL != pattern) {

		pattern1 = md_alloc(DIMS, pat1_dims, CFL_SIZE);
		md_slice(DIMS, loop_flags, (const long[DIMS]){ [0 ... DIMS - 1] = 0 }, pat_dims, pattern1, pattern, CFL_SIZE);
	}

	// FIXME: re-initialize forward_op and precond_op

	if ((NULL == traj_file) && ((0u != loop_flags && !sms) || conf.bpsense)) { // FIXME: no basis

		linop_free(forward_op);
		forward_op = sense_init(max1_dims, map_flags, maps);

		// basis pursuit requires the full forward model to add as a linop constraint
		if (conf.bpsense) {

			const struct linop_s* sample_op = linop_sampling_create(max1_dims, pat1_dims, pattern1);
			forward_op = linop_chain_FF(forward_op, sample_op);
		}
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


	opt_reg_configure(DIMS, img1_dims, &ropts, thresh_ops, trafos, llr_blk, shift_mode, wtype_str, conf.gpu);

	if (conf.bpsense)
		opt_bpursuit_configure(&ropts, thresh_ops, trafos, forward_op, kspace, bpsense_eps);
	
	if (conf.precond)
		opt_precond_configure(&ropts, thresh_ops, trafos, forward_op, DIMS, ksp_dims, kspace, pat_dims, conf.precond ? pattern : NULL);

	int nr_penalties = ropts.r + ropts.sr;

	debug_printf(DP_INFO, "Regularization terms: %d, Supporting variables: %d\n", nr_penalties, ropts.svars);

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

	struct iter it = italgo_config(algo, nr_penalties, ropts.regs, maxiter, step, hogwild, fast, admm, scaling, warm_start);

	if (eigen && (ALGO_PRIDU == algo))
		CAST_DOWN(iter_chambolle_pock_conf, it.iconf)->maxeigen_iter = 30;

	if (ALGO_CG == algo)
		nr_penalties = 0;

	bool trafos_cond = (   (ALGO_PRIDU == algo)
			    || (ALGO_ADMM == algo)
			    || (   (ALGO_NIHT == algo)
				&& (ropts.regs[0].xform == NIHTWAV)));

	// FIXME: will fail with looped dims
	struct iter_monitor_s* monitor = NULL;

	if (im_truth)
		monitor = iter_monitor_create(2 * md_calc_size(DIMS, img_dims), (const float*)image_truth, NULL, NULL);

	if (0 < ropts.svars) {

		assert(!im_truth);
		assert(!conf.rvc);
		assert(1 == img1_dims[BATCH_DIM]);
		assert(1 == max1_dims[BATCH_DIM]);
		assert(0 == loop_flags);

		long img2_dims[DIMS];
		md_copy_dims(DIMS, img2_dims, img1_dims);

		img2_dims[BATCH_DIM] += ropts.svars;
		long pos[DIMS] = { 0 };

		max1_dims[BATCH_DIM] += ropts.svars;

		forward_op = linop_chain_FF(
				linop_extract_create(DIMS, pos, linop_domain(forward_op)->dims, img2_dims),
				forward_op);
	}

	const struct operator_p_s* po = sense_recon_create(&conf, max1_dims, forward_op,
				pat1_dims, ((NULL != traj_file) || conf.bpsense) ? NULL : pattern1,
				it.italgo, it.iconf, image_start1, nr_penalties, thresh_ops,
				trafos_cond ? trafos : NULL, NULL, monitor);

	const struct operator_s* op = operator_p_bind(po, 1.);
	operator_p_free(po);

	if (0 < ropts.svars) {

		assert(img1_dims[BATCH_DIM] + ropts.svars == operator_codomain(op)->dims[BATCH_DIM]);

		long pos[DIMS] = { 0 };

		auto extr = linop_extract_create(DIMS, pos, img1_dims, operator_codomain(op)->dims);

		auto op2 = operator_chain(op, extr->forward);

		operator_free(op);
		op = op2;

		assert(img1_dims[BATCH_DIM] == operator_codomain(op)->dims[BATCH_DIM]);

		linop_free(extr);
	}

	long strsx[2][DIMS];
	const long* strs[2] = { strsx[0], strsx[1] };

	md_calc_strides(DIMS, strsx[0], img_dims, CFL_SIZE);
	md_calc_strides(DIMS, strsx[1], ksp_dims, CFL_SIZE);

	for (int i = 0; i < DIMS; i++) {

		if (MD_IS_SET(loop_flags, i)) {

			strsx[0][i] = 0;
			strsx[1][i] = 0;
		}
	}

	if (0 != loop_flags) {

		auto op_tmp = operator_copy_wrapper(2, strs, op);
		operator_free(op);
		op = op_tmp;

		// op = operator_loop(DIMS, loop_dims, op);
		op_tmp = operator_loop_parallel(DIMS, loop_dims, op, loop_flags, conf.gpu);
		operator_free(op);
		op = op_tmp;
	}

	operator_apply(op, DIMS, img_dims, image, DIMS, (conf.bpsense || conf.precond) ? img_dims : ksp_dims, (conf.bpsense || conf.precond) ? NULL : kspace);

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

	if (NULL != pattern1)
		md_free(pattern1);


	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, img_dims, image);

	if (NULL != traj)
		unmap_cfl(DIMS, traj_dims, traj);

	if (im_truth) {

#ifdef USE_CUDA
		if (conf.gpu)
			md_free(image_truth);
		else
#endif
			unmap_cfl(DIMS, img_dims, image_truth);
	}

	if (image_start)
		unmap_cfl(DIMS, img_dims, image_start);

	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

	return 0;
}

/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014-2016 Frank Ong <frankong@berkeley.edu>
 * 2014-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
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
#include "linops/sampling.h"
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

static const char usage_str[] = "<kspace> <sensitivities> <output>";
static const char help_str[] = "Parallel-imaging compressed-sensing reconstruction.";



static const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps, const long ksp_dims[DIMS], const long traj_dims[DIMS], const complex float* traj, struct nufft_conf_s conf, const long wgs_dims[DIMS], const complex float* weights, const long basis_dims[DIMS], const complex float* basis, struct operator_s** precond_op, bool sms)
{
	long coilim_dims[DIMS];
	long img_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

	long ksp_dims2[DIMS];
	md_copy_dims(DIMS, ksp_dims2, ksp_dims);
	ksp_dims2[COEFF_DIM] = max_dims[COEFF_DIM];
	//ksp_dims2[TE_DIM] = 1;

	debug_print_dims(DP_INFO, DIMS, ksp_dims2);
	debug_print_dims(DP_INFO, DIMS, coilim_dims);

	const struct linop_s* fft_op = nufft_create2(DIMS, ksp_dims2, coilim_dims, traj_dims, traj, wgs_dims, weights, basis_dims, basis, conf);
	const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims, img_dims, maps);

	if (sms) {

		/**
		 * Apply Fourier encoding in image space (after coil
		 * sensitivity weighting but before NUFFT).
		 */

		const struct linop_s* fft_slice = linop_fft_create(DIMS, coilim_dims, SLICE_FLAG);

		fft_op = linop_chain_FF(fft_slice, fft_op);
	}

	const struct linop_s* lop = linop_chain_FF(maps_op, fft_op);

	//precond_op[0] = (struct operator_s*) nufft_precond_create( fft_op );
	precond_op[0] = NULL;

	return lop;
}


int main_pics(int argc, char* argv[])
{
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
	nuconf.lowmem = true;

	float restrict_fov = -1.;
	const char* pat_file = NULL;
	const char* traj_file = NULL;
	bool scale_im = false;
	bool eigen = false;
	float scaling = 0.;

        // Simultaneous Multi-Slice
        bool sms = false;

	unsigned int llr_blk = 8;

	const char* image_truth_file = NULL;
	bool im_truth = false;

	const char* image_start_file = NULL;
	bool warm_start = false;

	const char* basis_file = NULL;

	struct admm_conf admm = { false, false, false, iter_admm_defaults.rho, iter_admm_defaults.maxitercg };

	enum algo_t algo = ALGO_DEFAULT;

	bool hogwild = false;
	bool fast = false;

	unsigned int gpun = 0;

	struct opt_reg_s ropts;
	opt_reg_init(&ropts);

	unsigned int loop_flags = 0u;

	const struct opt_s opts[] = {

		{ 'l', true, opt_reg, &ropts, "1/-l2\t\ttoggle l1-wavelet or l2 regularization." },
		OPT_FLOAT('r', &ropts.lambda, "lambda", "regularization parameter"),
		{ 'R', true, opt_reg, &ropts, " <T>:A:B:C\tgeneralized regularization options (-Rh for help)" },
		OPT_SET('c', &conf.rvc, "real-value constraint"),
		OPT_FLOAT('s', &step, "step", "iteration stepsize"),
		OPT_UINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_STRING('t', &traj_file, "file", "k-space trajectory"),
		OPT_CLEAR('n', &randshift, "disable random wavelet cycle spinning"),
		OPT_SET('N', &overlapping_blocks, "do fully overlapping LLR blocks"),
		OPT_SET('g', &conf.gpu, "use GPU"),
		OPT_UINT('G', &gpun, "gpun", "use GPU device gpun"),
		OPT_STRING('p', &pat_file, "file", "pattern or weights"),
		OPT_SELECT('I', enum algo_t, &algo, ALGO_IST, "select IST"),
		OPT_UINT('b', &llr_blk, "blk", "Lowrank block size"),
		OPT_SET('e', &eigen, "Scale stepsize based on max. eigenvalue"),
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_SET('D', &admm.dynamic_rho, "(ADMM dynamic step size)"),
		OPT_SET('F', &fast, "(fast)"),
		OPT_SET('J', &admm.relative_norm, "(ADMM residual balancing)"),
		OPT_STRING('T', &image_truth_file, "file", "(truth file)"),
		OPT_STRING('W', &image_start_file, "<img>", "Warm start with <img>"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_INT('O', &conf.rwiter, "rwiter", "(reweighting)"),
		OPT_FLOAT('o', &conf.gamma, "gamma", "(reweighting)"),
		OPT_FLOAT('u', &admm.rho, "rho", "ADMM rho"),
		OPT_UINT('C', &admm.maxitercg, "iter", "ADMM max. CG iterations"),
		OPT_FLOAT('q', &conf.cclambda, "cclambda", "(cclambda)"),
		OPT_FLOAT('f', &restrict_fov, "rfov", "restrict FOV"),
		OPT_SELECT('m', enum algo_t, &algo, ALGO_ADMM, "select ADMM"),
		OPT_FLOAT('w', &scaling, "val", "inverse scaling of the data"),
		OPT_SET('S', &scale_im, "re-scale the image after reconstruction"),
		OPT_UINT('L', &loop_flags, "flags", "batch-mode"),
		OPT_SET('K', &nuconf.pcycle, "randshift for NUFFT"),
		OPT_STRING('B', &basis_file, "file", "temporal (or other) basis"),
		OPT_FLOAT('P', &bpsense_eps, "eps", "Basis Pursuit formulation, || y- Ax ||_2 <= eps"),
		OPT_SELECT('a', enum algo_t, &algo, ALGO_PRIDU, "select Primal Dual"),
		OPT_SET('M', &sms, "Simultaneous Multi-Slice reconstruction"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (NULL != image_truth_file)
		im_truth = true;

	if (NULL != image_start_file)
		warm_start = true;

	if (0 <= bpsense_eps)
		conf.bpsense = true;

	admm.dynamic_tau = admm.relative_norm;

	if (conf.bpsense)
		nuconf.toeplitz = false;


	long max_dims[DIMS];
	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long coilim_dims[DIMS];
	long ksp_dims[DIMS];
	long traj_dims[DIMS];


	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(argv[1], DIMS, ksp_dims);

        if (sms) {

                debug_printf(DP_INFO, "SMS reconstruction: MB = %ld\n", ksp_dims[SLICE_DIM]);
        }

	complex float* maps = load_cfl(argv[2], DIMS, map_dims);

	unsigned int map_flags = md_nontriv_dims(DIMS, map_dims);

	map_flags |= FFT_FLAGS | SENS_FLAGS;



	long basis_dims[DIMS];
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

	assert(1 == ksp_dims[COEFF_DIM]);
	long bmx_dims[DIMS];

	if (NULL != basis_file) {

		assert(basis_dims[TE_DIM] == ksp_dims[TE_DIM]);

		max_dims[COEFF_DIM] = basis_dims[COEFF_DIM];

		md_copy_dims(DIMS, bmx_dims, max_dims);
		debug_printf(DP_INFO, "Basis: ");
		debug_print_dims(DP_INFO, DIMS, bmx_dims);

		max_dims[TE_DIM] = 1;

		debug_printf(DP_INFO, "Max:   ");
		debug_print_dims(DP_INFO, DIMS, max_dims);

//		if (NULL != traj_file)
//			nuconf.toeplitz = false;
	}


	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);

	if (!md_check_compat(DIMS, ~(MD_BIT(MAPS_DIM)|FFT_FLAGS), img_dims, map_dims))
		error("Dimensions of image and sensitivities do not match!\n");

	if ((NULL != traj_file) && (!md_check_compat(DIMS, ~0, ksp_dims, traj_dims)))
		error("Dimensions of data and trajectory do not match!\n");



	assert(1 == ksp_dims[MAPS_DIM]);


	if (conf.gpu)
		num_init_gpu_device(gpun);
	else
		num_init();

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

		if (NULL == pat_file && NULL == basis) {

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


	// initialize forward_op and precond_op

	const struct linop_s* forward_op = NULL;
	const struct operator_s* precond_op = NULL;

	if (NULL == traj_file) {

		forward_op = sense_init(max_dims, map_flags, maps);

		// apply temporal basis

		if (NULL != basis_file) {

			const struct linop_s* basis_op = linop_fmac_create(DIMS, bmx_dims, COEFF_FLAG, TE_FLAG, ~(COEFF_FLAG | TE_FLAG), basis);
			forward_op = linop_chain_FF(forward_op, basis_op);
		}

	} else {

		forward_op = sense_nc_init(max_dims, map_dims, maps, ksp_dims, traj_dims, traj, nuconf,
				pat_dims, pattern, basis_dims, basis, (struct operator_s**)&precond_op, sms);
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

	if (0. == scaling ) {

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


	complex float* image = create_cfl(argv[3], DIMS, img_dims);
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

		xfree(image_start_file);

		// if rescaling at the end, assume the input has also been rescaled
		if (scale_im && (scaling != 0.))
			md_zsmul(DIMS, img_dims, image_start, image_start, 1. /  scaling);
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

	if ((NULL == traj_file) && (0u != loop_flags) && !sms) { // FIXME: no basis

		linop_free(forward_op);
		forward_op = sense_init(max1_dims, map_flags, maps);

		// basis pursuit requires the full forward model to add as a linop constraint
		if (conf.bpsense) {

			const struct linop_s* sample_op = linop_sampling_create(max1_dims, pat1_dims, pattern1);
			struct linop_s* tmp = linop_chain(forward_op, sample_op);

			linop_free(sample_op);
			linop_free(forward_op);

			forward_op = tmp;
		}
	}

	double maxeigen = 1.;

	if (eigen) {

		maxeigen = estimate_maxeigenval(forward_op->normal);

		debug_printf(DP_INFO, "Maximum eigenvalue: %.2e\n", maxeigen);

	}


	// initialize prox functions

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	opt_reg_configure(DIMS, img1_dims, &ropts, thresh_ops, trafos, llr_blk, shift_mode, conf.gpu);

	if (conf.bpsense)
		opt_bpursuit_configure(&ropts, thresh_ops, trafos, forward_op, kspace, bpsense_eps);

	int nr_penalties = ropts.r;
	struct reg_s* regs = ropts.regs;

	// choose algorithm

	if (ALGO_DEFAULT == algo)
		algo = italgo_choose(nr_penalties, regs);

	if (conf.bpsense)
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

	struct iter it = italgo_config(algo, nr_penalties, regs, maxiter, step, hogwild, fast, admm, scaling, warm_start);

	if (ALGO_CG == algo)
		nr_penalties = 0;

	bool trafos_cond = (   (ALGO_PRIDU == algo)
			    || (ALGO_ADMM == algo)
			    || (   (ALGO_NIHT == algo)
				&& (regs[0].xform == NIHTWAV)));

	// FIXME: will fail with looped dims
	struct iter_monitor_s* monitor = NULL;
	if (im_truth)
		monitor = create_monitor(2*md_calc_size(DIMS, img_dims), (const float*)image_truth, NULL, NULL); 
	
	const struct operator_p_s* po = sense_recon_create(&conf, max1_dims, forward_op,
				pat1_dims, ((NULL != traj_file) || conf.bpsense) ? NULL : pattern1,
				it.italgo, it.iconf, image_start1, nr_penalties, thresh_ops,
				trafos_cond ? trafos : NULL, precond_op, monitor);

	const struct operator_s* op = operator_p_bind(po, 1.);
//	operator_p_free(po);	// FIXME

	long strsx[2][DIMS];
	const long* strs[2] = { strsx[0], strsx[1] };

	md_calc_strides(DIMS, strsx[0], img_dims, CFL_SIZE);
	md_calc_strides(DIMS, strsx[1], ksp_dims, CFL_SIZE);

	for (unsigned int i = 0; i < DIMS; i++) {

		if (MD_IS_SET(loop_flags, i)) {

			strsx[0][i] = 0;
			strsx[1][i] = 0;
		}
	}

	if (0 != loop_flags) {

		op = operator_copy_wrapper(2, strs, op);
		// op = operator_loop(DIMS, loop_dims, op);
		op = operator_loop_parallel(DIMS, loop_dims, op, loop_flags, conf.gpu);
	}

	operator_apply(op, DIMS, img_dims, image, DIMS, conf.bpsense ? img_dims : ksp_dims, conf.bpsense ? NULL : kspace);

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

	xfree(pat_file);
	xfree(traj_file);
	xfree(basis_file);

	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);

	return 0;
}



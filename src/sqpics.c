/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014-2016 Frank Ong <frankong@berkeley.edu>
 * 2014-2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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
#include "num/ops.h"
#include "num/iovec.h"

#include "iter/prox.h"
#include "iter/thresh.h"
#include "iter/misc.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/sampling.h"

#include "iter/iter.h"
#include "iter/iter2.h"

#include "noncart/nufft.h"

//#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "wavelet3/wavthresh.h"

#include "lowrank/lrthresh.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#define NUM_REGS 10

static const char usage_str[] = "<kspace> <sensitivities> <output>";
static const char help_str[] = "Parallel-imaging compressed-sensing reconstruction.";

static void help_reg(void)
{
	printf( "Generalized regularization options (experimental)\n\n"
		"-R <T>:A:B:C\t<T> is regularization type (single letter),\n"
		"\t\tA is transform flags, B is joint threshold flags,\n"
		"\t\tand C is regularization value. Specify any number\n"
		"\t\tof regularization terms.\n\n"
		"-R Q:C    \tl2-norm in image domain\n"
		"-R I:B:C  \tl1-norm in image domain\n"
		"-R W:A:B:C\tl1-wavelet\n"
		"-R T:A:B:C\ttotal variation\n"
		"-R T:7:0:.01\t3D isotropic total variation with 0.01 regularization.\n"
		"-R L:7:7:.02\tLocally low rank with spatial decimation and 0.02 regularization.\n"
		"-R M:7:7:.03\tMulti-scale low rank with spatial decimation and 0.03 regularization.\n"
	);
}

 
static
const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps, const long ksp_dims[DIMS], const long traj_dims[DIMS], const complex float* traj, struct nufft_conf_s conf, struct operator_s** precond_op)
{
	long coilim_dims[DIMS];
	long img_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);
	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

	const struct linop_s* fft_op = nufft_create(DIMS, ksp_dims, coilim_dims, traj_dims, traj, NULL, conf);
	const struct linop_s* maps_op = maps2_create(coilim_dims, map_dims, img_dims, maps);

	//precond_op[0] = (struct operator_s*) nufft_precond_create( fft_op );
	precond_op[0] = NULL;

	const struct linop_s* lop = linop_chain(maps_op, fft_op);

	linop_free(maps_op);
	linop_free(fft_op);

	return lop;
}

struct reg_s {

	enum { L1WAV, TV, LLR, MLR, IMAGL1, IMAGL2, L1IMG, L2IMG } xform;

	unsigned int xflags;
	unsigned int jflags;

	float lambda;
};

enum algo_t { CG, IST, FISTA, ADMM };

struct opt_reg_s {

	float lambda;
	enum algo_t algo;
	struct reg_s regs[NUM_REGS];
	unsigned int r;
};

static bool opt_reg(void* ptr, char c, const char* optarg)
{
	struct opt_reg_s* p = ptr;
	struct reg_s* regs = p->regs;
	const int r = p->r;
	const float lambda = p->lambda;

	assert(r < NUM_REGS);

	char rt[5];

	switch (c) {

	case 'R': {

		// first get transform type
		int ret = sscanf(optarg, "%4[^:]", rt);
		assert(1 == ret);

		// next switch based on transform type
		if (strcmp(rt, "W") == 0) {

			regs[r].xform = L1WAV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "L") == 0) {

			regs[r].xform = LLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "M") == 0) {

			regs[r].xform = regs[0].xform;
			regs[r].xflags = regs[0].xflags;
			regs[r].jflags = regs[0].jflags;
			regs[r].lambda = regs[0].lambda;

			regs[0].xform = MLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[0].xflags, &regs[0].jflags, &regs[0].lambda);
			assert(3 == ret);
		}
		else if (strcmp(rt, "T") == 0) {

			regs[r].xform = TV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
			p->algo = ADMM;
		}
		else if (strcmp(rt, "R1") == 0) {

			regs[r].xform = IMAGL1;
			int ret = sscanf(optarg, "%*[^:]:%d:%f", &regs[r].jflags, &regs[r].lambda);
			assert(2 == ret);
			regs[r].xflags = 0u;
			p->algo = ADMM;
		}
		else if (strcmp(rt, "R2") == 0) {

			regs[r].xform = IMAGL2;
			int ret = sscanf(optarg, "%*[^:]:%d:%f", &regs[r].jflags, &regs[r].lambda);
			assert(2 == ret);
			regs[r].xflags = 0u;
			p->algo = ADMM;
		}
		else if (strcmp(rt, "I") == 0) {

			regs[r].xform = L1IMG;
			int ret = sscanf(optarg, "%*[^:]:%d:%f", &regs[r].jflags, &regs[r].lambda);
			assert(2 == ret);
			regs[r].xflags = 0u;
		}
		else if (strcmp(rt, "Q") == 0) {

			regs[r].xform = L2IMG;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;
		}
		else if (strcmp(rt, "h") == 0) {

			help_reg();
			exit(0);
		}
		else {

			error("Unrecognized regularization type: \"%s\" (-Rh for help).\n", rt);
		}

		p->r++;
		break;
	}

	case 'l':
		assert(r < NUM_REGS);
		regs[r].lambda = lambda;
		regs[r].xflags = 0u;
		regs[r].jflags = 0u;

		if (0 == strcmp("1", optarg)) {

			regs[r].xform = L1WAV;
			regs[r].xflags = 7u;

		} else
		if (0 == strcmp("2", optarg)) {

			regs[r].xform = L2IMG;

		} else {

			error("Unknown regularization type.\n");
		}

		p->lambda = -1.;
		p->r++;
		break;
	}

	return false;
}

int main_sqpics(int argc, char* argv[])
{
	// Initialize default parameters


	bool use_gpu = false;

	bool randshift = true;
	unsigned int maxiter = 30;
	float step = -1.;

	// Start time count

	double start_time = timestamp();

	// Read input options
	struct nufft_conf_s nuconf = nufft_conf_defaults;
	nuconf.toeplitz = false;

	float restrict_fov = -1.;
	const char* pat_file = NULL;
	const char* traj_file = NULL;
	bool scale_im = false;
	bool eigen = false;
	float scaling = 0.;

	unsigned int llr_blk = 8;

	const char* image_truth_file = NULL;
	bool im_truth = false;

	const char* image_start_file = NULL;
	bool warm_start = false;

	bool hogwild = false;
	bool fast = false;
	float admm_rho = iter_admm_defaults.rho;
	unsigned int admm_maxitercg = iter_admm_defaults.maxitercg;

	struct opt_reg_s ropts;
	ropts.r = 0;
	ropts.algo = CG;
	ropts.lambda = -1.;


	const struct opt_s opts[] = {

		{ 'l', true, opt_reg, &ropts, "1/-l2\t\ttoggle l1-wavelet or l2 regularization." },
		OPT_FLOAT('r', &ropts.lambda, "lambda", "regularization parameter"),
		{ 'R', true, opt_reg, &ropts, " <T>:A:B:C\tgeneralized regularization options (-Rh for help)" },
		//OPT_SET('c', &conf.rvc, "real-value constraint"),
		OPT_FLOAT('s', &step, "step", "iteration stepsize"),
		OPT_UINT('i', &maxiter, "iter", "max. number of iterations"),
		OPT_STRING('t', &traj_file, "file", "k-space trajectory"),
		OPT_CLEAR('n', &randshift, "disable random wavelet cycle spinning"),
		OPT_SET('g', &use_gpu, "use GPU"),
		OPT_STRING('p', &pat_file, "file", "pattern or weights"),
		OPT_SELECT('I', enum algo_t, &ropts.algo, IST, "(select IST)"),
		OPT_UINT('b', &llr_blk, "blk", "Lowrank block size"),
		OPT_SET('e', &eigen, "Scale stepsize based on max. eigenvalue"),
		OPT_SET('H', &hogwild, "(hogwild)"),
		OPT_SET('F', &fast, "(fast)"),
		OPT_STRING('T', &image_truth_file, "file", "(truth file)"),
		OPT_STRING('W', &image_start_file, "<img>", "Warm start with <img>"),
		OPT_INT('d', &debug_level, "level", "Debug level"),
		OPT_FLOAT('u', &admm_rho, "rho", "ADMM rho"),
		OPT_UINT('C', &admm_maxitercg, "iter", "ADMM max. CG iterations"),
		OPT_FLOAT('f', &restrict_fov, "rfov", "restrict FOV"),
		OPT_SELECT('m', enum algo_t, &ropts.algo, ADMM, "Select ADMM"),
		OPT_FLOAT('w', &scaling, "val", "scaling"),
		OPT_SET('S', &scale_im, "Re-scale the image after reconstruction"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (NULL != image_truth_file)
		im_truth = true;

	if (NULL != image_start_file)
		warm_start = true;


	long max_dims[DIMS];
	long map_dims[DIMS];
	long pat_dims[DIMS];
	long img_dims[DIMS];
	long coilim_dims[DIMS];
	long ksp_dims[DIMS];
	long traj_dims[DIMS];



	// load kspace and maps and get dimensions

	complex float* kspace = load_cfl(argv[1], DIMS, ksp_dims);
	complex float* maps = load_cfl(argv[2], DIMS, map_dims);


	complex float* traj = NULL;

	if (NULL != traj_file)
		traj = load_cfl(traj_file, DIMS, traj_dims);


	md_copy_dims(DIMS, max_dims, ksp_dims);
	md_copy_dims(5, max_dims, map_dims);

	md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, coilim_dims, max_dims);

	if (!md_check_compat(DIMS, ~(MD_BIT(MAPS_DIM)|FFT_FLAGS), img_dims, map_dims))
		error("Dimensions of image and sensitivities do not match!\n");

	assert(1 == ksp_dims[MAPS_DIM]);


	(use_gpu ? num_init_gpu : num_init)();

	// print options

	if (use_gpu)
		debug_printf(DP_INFO, "GPU reconstruction\n");

	if (map_dims[MAPS_DIM] > 1) 
		debug_printf(DP_INFO, "%ld maps.\nESPIRiT reconstruction.\n", map_dims[MAPS_DIM]);

	if (hogwild)
		debug_printf(DP_INFO, "Hogwild stepsize\n");

	if (im_truth)
		debug_printf(DP_INFO, "Compare to truth\n");



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


	if ((NULL != traj_file) && (NULL == pat_file)) {

		md_free(pattern);
		pattern = NULL;
		nuconf.toeplitz = true;

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

	if (NULL == traj_file)
		forward_op = sense_init(max_dims, FFT_FLAGS|COIL_FLAG|MAPS_FLAG, maps);
	else
		forward_op = sense_nc_init(max_dims, map_dims, maps, ksp_dims, traj_dims, traj, nuconf, (struct operator_s**) &precond_op);

	// apply scaling

	if (scaling == 0.) {

		if (NULL == traj_file) {

			scaling = estimate_scaling(ksp_dims, NULL, kspace);

		} else {

			complex float* adj = md_alloc(DIMS, img_dims, CFL_SIZE);

			linop_adjoint(forward_op, DIMS, img_dims, adj, DIMS, ksp_dims, kspace);
			scaling = estimate_scaling_norm(1., md_calc_size(DIMS, img_dims), adj, false);

			md_free(adj);
		}
	}
	else
		debug_printf(DP_DEBUG1, "Scaling: %f\n", scaling);

	if (scaling != 0.)
		md_zsmul(DIMS, ksp_dims, kspace, kspace, 1. / scaling);

	float lambda = ropts.lambda;

	if (-1. == lambda)
		lambda = 0.;

	// if no penalities specified but regularization
	// parameter is given, add a l2 penalty

	struct reg_s* regs = ropts.regs;

	if ((0 == ropts.r) && (lambda > 0.)) {

		regs[0].xform = L2IMG;
		regs[0].xflags = 0u;
		regs[0].jflags = 0u;
		regs[0].lambda = lambda;
		ropts.r = 1;
	}



	// initialize thresh_op
	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };
	int nr_penalties = ropts.r;
	long blkdims[MAX_LEV][DIMS];
	int levels;


	for (int nr = 0; nr < nr_penalties; nr++) {

		// fix up regularization parameter
		if (-1. == regs[nr].lambda)
			regs[nr].lambda = lambda;

		switch (regs[nr].xform) {

		case L1WAV:
			debug_printf(DP_INFO, "l1-wavelet regularization: %f\n", regs[nr].lambda);

			if (0 != regs[nr].jflags)
				debug_printf(DP_WARN, "joint l1-wavelet thresholding not currently supported.\n");

			long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
			minsize[0] = MIN(img_dims[0], 16);
			minsize[1] = MIN(img_dims[1], 16);
			minsize[2] = MIN(img_dims[2], 16);

			unsigned int wflags = 0;

			for (unsigned int i = 0; i < DIMS; i++) {

				if ((1 < img_dims[i]) && MD_IS_SET(regs[nr].xflags, i)) {

					wflags = MD_SET(wflags, i);
					minsize[i] = MIN(img_dims[i], 16);
				}
			}

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			thresh_ops[nr] = prox_wavelet3_thresh_create(DIMS, img_dims, wflags, minsize, regs[nr].lambda, randshift);
			break;

		case TV:
			debug_printf(DP_INFO, "TV regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_grad_create(DIMS, img_dims, regs[nr].xflags);
			thresh_ops[nr] = prox_thresh_create(DIMS + 1,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS), use_gpu);
			break;

		case LLR:
			debug_printf(DP_INFO, "lowrank regularization: %f\n", regs[nr].lambda);

			// add locally lowrank penalty
			levels = llr_blkdims(blkdims, regs[nr].jflags, img_dims, llr_blk);

			assert(1 == levels);
			img_dims[LEVEL_DIM] = levels;

			for(int l = 0; l < levels; l++)
#if 0
				blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];
#else
				blkdims[l][MAPS_DIM] = 1;
#endif

			int remove_mean = 0;

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			thresh_ops[nr] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, remove_mean, use_gpu);
			break;
       
		case MLR:
			debug_printf(DP_INFO, "multi-scale lowrank regularization: %f\n", regs[nr].lambda);

                        levels = multilr_blkdims(blkdims, regs[nr].jflags, img_dims, 8, 1);

			img_dims[LEVEL_DIM] = levels;
                        max_dims[LEVEL_DIM] = levels;

			for(int l = 0; l < levels; l++)
				blkdims[l][MAPS_DIM] = 1;

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			thresh_ops[nr] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, 0, use_gpu);

			const struct linop_s* decom_op = linop_sum_create(img_dims, use_gpu);
			const struct linop_s* tmp_op = forward_op;
			forward_op = linop_chain(decom_op, forward_op);

			linop_free(decom_op);
			linop_free(tmp_op);

			break;

		case IMAGL1:
			debug_printf(DP_INFO, "l1 regularization of imaginary part: %f\n", regs[nr].lambda);

			trafos[nr] = linop_rdiag_create(DIMS, img_dims, 0, &(complex float){ 1.i });
			thresh_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags, use_gpu);
			break;

		case IMAGL2:
			debug_printf(DP_INFO, "l2 regularization of imaginary part: %f\n", regs[nr].lambda);

			trafos[nr] = linop_rdiag_create(DIMS, img_dims, 0, &(complex float){ 1.i });
			thresh_ops[nr] = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);
			break;

		case L1IMG:
			debug_printf(DP_INFO, "l1 regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			thresh_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags, use_gpu);
			break;

		case L2IMG:
			debug_printf(DP_INFO, "l2 regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			thresh_ops[nr] = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);
			break;
		}

	}

	int nr = nr_penalties;
	struct linop_s* sampling = linop_sampling_create(max_dims, pat_dims, pattern);
	struct linop_s* tmp_op = linop_chain(forward_op, sampling);
	linop_free(sampling);
	linop_free(forward_op);
	forward_op = tmp_op;
	trafos[nr] = forward_op;
	thresh_ops[nr] = prox_l2norm_create(DIMS, ksp_dims, 1.);
	nr_penalties++;
	const float** biases = xmalloc(sizeof(float*) * nr_penalties);
	for (int i = 0; i < nr_penalties - 1; i++)
		biases[i] = NULL;

	biases[nr] = (float*)kspace;



	complex float* image = create_cfl(argv[3], DIMS, img_dims);
	md_clear(DIMS, img_dims, image, CFL_SIZE);


	long img_truth_dims[DIMS];
	complex float* image_truth = NULL;

	if (im_truth) {

		image_truth = load_cfl(image_truth_file, DIMS, img_truth_dims);
		//md_zsmul(DIMS, img_dims, image_truth, image_truth, 1. / scaling);
	}

	long img_start_dims[DIMS];
	complex float* image_start = NULL;

	if (warm_start) { 

		debug_printf(DP_DEBUG1, "Warm start: %s\n", image_start_file);
		image_start = load_cfl(image_start_file, DIMS, img_start_dims);
		assert(md_check_compat(DIMS, 0u, img_start_dims, img_dims));
		md_copy(DIMS, img_dims, image, image_start, CFL_SIZE);

		free((void*)image_start_file);
		unmap_cfl(DIMS, img_dims, image_start);

		// if rescaling at the end, assume the input has also been rescaled
		if (scale_im && scaling != 0.)
			md_zsmul(DIMS, img_dims, image, image, 1. /  scaling);
	}


	// initialize algorithm

	struct iter_admm_conf mmconf;


	debug_printf(DP_INFO, "ADMM\n");

	mmconf = iter_admm_defaults;
	mmconf.maxiter = maxiter;
	mmconf.maxitercg = admm_maxitercg;
	mmconf.rho = admm_rho;
	mmconf.hogwild = hogwild;
	mmconf.fast = fast;
	//mmconf.dynamic_rho = true;
	mmconf.ABSTOL = 0.;
	mmconf.RELTOL = 0.;


	long size = 2 * md_calc_size(DIMS, img_dims);
	iter2_admm(CAST_UP(&mmconf), NULL, nr_penalties, thresh_ops, trafos, biases, NULL, size, (float*)image, NULL, NULL);


#if 0
	if (use_gpu) 
#ifdef USE_CUDA
		sqpics_recon2_gpu(&conf, max_dims, image, forward_op, pat_dims, pattern,
				 italgo, iconf, nr_penalties, thresh_ops,
				 (ADMM == algo) ? trafos : NULL, ksp_dims, kspace, image_truth, precond_op);
#else
	assert(0);
#endif
	else
		sqpics_recon2(&conf, max_dims, image, forward_op, pat_dims, pattern,
			     italgo, iconf, nr_penalties, thresh_ops,
			     (ADMM == algo) ? trafos : NULL, ksp_dims, kspace, image_truth, precond_op);
#endif

	if (scale_im)
		md_zsmul(DIMS, img_dims, image, image, scaling);

	// clean up

	if (NULL != pat_file)
		unmap_cfl(DIMS, pat_dims, pattern);
	else
		md_free(pattern);


	unmap_cfl(DIMS, map_dims, maps);
	unmap_cfl(DIMS, ksp_dims, kspace);
	unmap_cfl(DIMS, img_dims, image);

	if (NULL != traj)
		unmap_cfl(DIMS, traj_dims, traj);

	if (im_truth) {

		free((void*)image_truth_file);
		unmap_cfl(DIMS, img_dims, image_truth);
	}


	double end_time = timestamp();

	debug_printf(DP_INFO, "Total Time: %f\n", end_time - start_time);
	exit(0);
}



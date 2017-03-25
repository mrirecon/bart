/* Copyright 2015-2016. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015-2016 Frank Ong <frankong@berkeley.edu>
 * 2015-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "num/multind.h"
#include "num/iovec.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/grad.h"
#include "linops/sum.h"

#include "wavelet3/wavthresh.h"

#include "lowrank/lrthresh.h"

#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "optreg.h"




void help_reg(void)
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




bool opt_reg(void* ptr, char c, const char* optarg)
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
		else if (strcmp(rt, "P") == 0) {

			regs[r].xform = LAPLACE;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
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
		else if (strcmp(rt, "F") == 0) {

			regs[r].xform = FTL1;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);
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

		} else if (0 == strcmp("2", optarg)) {

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

bool opt_reg_init(struct opt_reg_s* ropts)
{
	ropts->r = 0;
	ropts->algo = CG;
	ropts->lambda = -1;

	return false;
}


void opt_reg_configure(unsigned int N, const long img_dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int llr_blk, bool randshift, bool use_gpu)
{
	float lambda = ropts->lambda;

	if (-1. == lambda)
		lambda = 0.;

	// if no penalities specified but regularization
	// parameter is given, add a l2 penalty

	struct reg_s* regs = ropts->regs;

	if ((0 == ropts->r) && (lambda > 0.)) {

		regs[0].xform = L2IMG;
		regs[0].xflags = 0u;
		regs[0].jflags = 0u;
		regs[0].lambda = lambda;
		ropts->r = 1;
	}



	int nr_penalties = ropts->r;
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
			prox_ops[nr] = prox_wavelet3_thresh_create(DIMS, img_dims, wflags, minsize, regs[nr].lambda, randshift);
			break;

		case TV:
			debug_printf(DP_INFO, "TV regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_grad_create(DIMS, img_dims, regs[nr].xflags);
			prox_ops[nr] = prox_thresh_create(DIMS + 1,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(DIMS), use_gpu);
			break;

		case LAPLACE:
			debug_printf(DP_INFO, "L1-Laplace regularization: %f\n", regs[nr].lambda);
			long krn_dims[DIMS] = { [0 ... DIMS - 1] = 1 };

			for (unsigned int i = 0; i < DIMS; i++)
				if (MD_IS_SET(regs[nr].xflags, i))
					krn_dims[i] = 3;

			complex float krn[] = {	// laplace filter
				-1., -2., -1.,
				-2., 12., -2.,
				-1., -2., -1.,
			};

			assert(9 == md_calc_size(DIMS, krn_dims));

			trafos[nr] = linop_conv_create(DIMS, regs[nr].xflags, CONV_SYMMETRIC, CONV_TRUNCATED, img_dims, img_dims, krn_dims, krn);
			prox_ops[nr] = prox_thresh_create(DIMS,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags, use_gpu);
			break;

		case LLR:

			debug_printf(DP_INFO, "lowrank regularization: %f\n", regs[nr].lambda);

			// add locally lowrank penalty
			levels = llr_blkdims(blkdims, regs[nr].jflags, img_dims, llr_blk);

			assert(1 == levels);

			assert(levels == img_dims[LEVEL_DIM]);

			for(int l = 0; l < levels; l++)
#if 0
				blkdims[l][MAPS_DIM] = img_dims[MAPS_DIM];
#else
				blkdims[l][MAPS_DIM] = 1;
#endif

			int remove_mean = 0;

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, remove_mean, use_gpu);
			break;

		case MLR:
#if 0
			// FIXME: multiscale low rank changes the output image dimensions 
			// and requires the forward linear operator. This should be decoupled...
			debug_printf(DP_INFO, "multi-scale lowrank regularization: %f\n", regs[nr].lambda);

			levels = multilr_blkdims(blkdims, regs[nr].jflags, img_dims, 8, 1);

			img_dims[LEVEL_DIM] = levels;
			max_dims[LEVEL_DIM] = levels;

			for(int l = 0; l < levels; l++)
				blkdims[l][MAPS_DIM] = 1;

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = lrthresh_create(img_dims, randshift, regs[nr].xflags, (const long (*)[DIMS])blkdims, regs[nr].lambda, false, 0, use_gpu);

			const struct linop_s* decom_op = sum_create( img_dims, use_gpu );
			const struct linop_s* tmp_op = forward_op;
			forward_op = linop_chain(decom_op, forward_op);

			linop_free(decom_op);
			linop_free(tmp_op);
#else
			debug_printf(DP_WARN, "multi-scale lowrank regularization not yet supported: %f\n", regs[nr].lambda);
#endif

			break;

		case IMAGL1:
			debug_printf(DP_INFO, "l1 regularization of imaginary part: %f\n", regs[nr].lambda);

			trafos[nr] = linop_rdiag_create(DIMS, img_dims, 0, &(complex float){ 1.i });
			prox_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags, use_gpu);
			break;

		case IMAGL2:
			debug_printf(DP_INFO, "l2 regularization of imaginary part: %f\n", regs[nr].lambda);

			trafos[nr] = linop_rdiag_create(DIMS, img_dims, 0, &(complex float){ 1.i });
			prox_ops[nr] = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);
			break;

		case L1IMG:
			debug_printf(DP_INFO, "l1 regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags, use_gpu);
			break;

		case L2IMG:
			debug_printf(DP_INFO, "l2 regularization: %f\n", regs[nr].lambda);

			trafos[nr] = linop_identity_create(DIMS, img_dims);
			prox_ops[nr] = prox_leastsquares_create(DIMS, img_dims, regs[nr].lambda, NULL);
			break;

		case FTL1:
			debug_printf(DP_INFO, "l1 regularization of Fourier transform: %f\n", regs[nr].lambda);

			trafos[nr] = linop_fft_create(DIMS, img_dims, regs[nr].xflags);
			prox_ops[nr] = prox_thresh_create(DIMS, img_dims, regs[nr].lambda, regs[nr].jflags, use_gpu);
			break;
		}

	}
}


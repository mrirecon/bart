/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 * 2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 * 
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>

#include "grecon/optreg.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "linops/grad.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/utils.h"

#include "num/multind.h"
#include "num/iovec.h"
#include "num/ops_p.h"
#include "num/ops.h"

#include "wavelet/wavthresh.h"
#include "lowrank/lrthresh.h"

#include "moba/meco.h"

#include "optreg.h"


static const struct operator_p_s* create_wav_prox(const long img_dims[DIMS], unsigned int jt_flag, float lambda)
{
	bool randshift = true;
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	unsigned int wflags = 0;

	for (unsigned int i = 0; i < DIMS; i++) {

		if ((1 < img_dims[i]) && MD_IS_SET(FFT_FLAGS, i)) {

			wflags = MD_SET(wflags, i);
			minsize[i] = MIN(img_dims[i], DIMS);
		}
	}

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, jt_flag, minsize, lambda, randshift);
}

static const struct operator_p_s* create_llr_prox(const long img_dims[DIMS], unsigned int jt_flag, float lambda)
{
	bool randshift = true;
	long blk_dims[MAX_LEV][DIMS];
	int blk_size = 16;

	int levels = llr_blkdims(blk_dims, ~jt_flag, img_dims, blk_size);
	UNUSED(levels);

	return lrthresh_create(img_dims, randshift, ~jt_flag, (const long (*)[])blk_dims, lambda, false, false, false);
}

static const struct operator_p_s* ops_p_stack_higher_dims(unsigned int N, const long maps_dims[N], unsigned int coeff_dim, long higher_flag, const struct operator_p_s* src)
{
	const struct operator_p_s* tmp = operator_p_ref(src);
	const struct operator_p_s* dst = operator_p_ref(src);

	for (long d = coeff_dim+1; d < N; d++) {

		if (MD_IS_SET(higher_flag, d)) {

			for (long p = 1; p < maps_dims[d]; p++)
				dst = operator_p_stack(d, d, dst, tmp);
			
			tmp = operator_p_ref(dst);
		}
	}

	operator_p_free(tmp);

	return dst;
}

static const struct operator_p_s* create_stack_spatial_thresh_prox(unsigned int N, const long x_dims[N], long js_dim, unsigned int regu, float lambda, unsigned int model)
{
	assert(MECO_PI != model);

	unsigned int wgh_fB0 = (MECO_PHASEDIFF == model) ? MECO_IDENTITY : MECO_SOBOLEV; // FIXME: this is hard-coded

	long nr_coeff = set_num_of_coeff(model);
	long D = x_dims[js_dim];

	long x_prox1_dims[N];
	md_copy_dims(N, x_prox1_dims, x_dims);
	x_prox1_dims[js_dim] = nr_coeff - 1; // exclude fB0

	long x_prox3_dims[N];
	md_copy_dims(N, x_prox3_dims, x_dims);
	x_prox3_dims[js_dim] = 1; // fB0

	long x_prox4_dims[N];
	md_copy_dims(N, x_prox4_dims, x_dims);
	x_prox4_dims[js_dim] = D - nr_coeff;

	debug_printf(DP_DEBUG4, " >> x_prox1_dims: ");
	debug_print_dims(DP_DEBUG4, N, x_prox1_dims);

	debug_printf(DP_DEBUG4, " >> x_prox3_dims: ");
	debug_print_dims(DP_DEBUG4, N, x_prox3_dims);

	const struct operator_p_s* pcurr = NULL;

	auto prox1 = ((L1WAV == regu) ? create_wav_prox : create_llr_prox)(x_prox1_dims, MD_BIT(js_dim), lambda);

	auto prox3 = prox_zero_create(N, x_prox3_dims);

	if (MECO_IDENTITY == wgh_fB0)
		prox3 = ((L1WAV == regu) ? create_wav_prox : create_llr_prox)(x_prox3_dims, MD_BIT(js_dim), lambda);

	auto prox4 = prox_zero_create(N, x_prox4_dims);
#if 0
	auto prox2 = op_p_auto_normalize(prox1, ~MD_BIT(js_dim));
	pcurr = operator_p_stack(js_dim, js_dim, prox2, prox3);

	operator_p_free(prox2);
	operator_p_free(prox3);
#else
	pcurr = operator_p_stack(js_dim, js_dim, prox1, prox3);

	operator_p_free(prox1);
	operator_p_free(prox3);
#endif

	pcurr = operator_p_stack(js_dim, js_dim, pcurr, prox4);

	operator_p_free(prox4);

	return pcurr;
}


const struct operator_p_s* moba_nonneg_prox_create(unsigned int N, const long maps_dims[N], unsigned int coeff_dim, long nonneg_flag, float lambda)
{
	// higher dimensions
	long higher_flag = 0;
	for (long d = coeff_dim+1; d < N; d++) {

		if (1 < maps_dims[d])
			higher_flag = MD_SET(higher_flag, d);
	}

	// single map dimensions
	long map_dims[N];
	md_select_dims(N, ~(MD_BIT(coeff_dim)|higher_flag), map_dims, maps_dims);

	const struct operator_p_s* p1 = prox_zsmax_create(N, map_dims, lambda);
	const struct operator_p_s* p2 = prox_zero_create(N, map_dims);
	const struct operator_p_s* p3 = NULL;

	const struct operator_p_s* prox_j = NULL;

	for (long m = 0; m < maps_dims[coeff_dim]; m++) {

		p3 = MD_IS_SET(nonneg_flag, m) ? p1 : p2;
		prox_j = (NULL == prox_j) ? p3 : operator_p_stack(coeff_dim, coeff_dim, prox_j, p3);
	}

	operator_p_free(p1);
	operator_p_free(p2);
	operator_p_free(p3);

	// stack higher dimensions
	auto prox_s = ops_p_stack_higher_dims(N, maps_dims, coeff_dim, higher_flag, prox_j);

	operator_p_free(prox_j);

	return prox_s;
}


static const struct operator_p_s* moba_sens_prox_create(unsigned int N, const long sens_dims[N])
{
	const struct operator_p_s* p = prox_zero_create(N, sens_dims);
	return p;
}

static const struct operator_p_s* flatten_prox(const struct operator_p_s* src)
{
	const struct operator_p_s* dst = operator_p_reshape_in_F(src, 1, MD_DIMS(md_calc_size(operator_p_domain(src)->N, operator_p_domain(src)->dims)));

	dst = operator_p_reshape_out_F(dst, 1, MD_DIMS(md_calc_size(operator_p_codomain(dst)->N, operator_p_codomain(dst)->dims)));

	return dst;
}

static const struct operator_p_s* stack_flatten_prox(const struct operator_p_s* prox_maps, const struct operator_p_s* prox_sens)
{
	auto prox1 = flatten_prox(prox_maps);
	auto prox2 = flatten_prox(prox_sens);

	auto prox3 = operator_p_stack(0, 0, prox1, prox2);

	operator_p_free(prox1);
	operator_p_free(prox2);

	return prox3;
}


void help_reg_moba(void)
{
	printf( "Generalized regularization options for model-based reconstructions (experimental)\n\n"
			"-R <T>:A:B:C\t<T> is regularization type (single letter),\n"
			"\t\tA is transform flags,\n"
			"\t\tB is joint threshold flags,\n"
			"\t\tC is regularization value.\n"
			"\t\tSpecify any number of regularization terms.\n\n"
			"-R W:0:0:C\tl1-wavelet (A and B are internally determined by moba models)\n"
			"-R L:0:0:C\tlocally low rank (A and B are internally determined by moba models)\n"
			"-R Q:C\tl2 regularization\n"
			"-R S:C\tnon-negative constraint\n");
}

bool opt_reg_moba(void* ptr, char c, const char* optarg)
{
	struct opt_reg_s* p = ptr;

	struct reg_s* regs = p->regs;
	const int r = p->r;

	assert(r < NUM_REGS);
	
	char rt[5];
	int ret;
	
	switch (c) {

	case 'r':

		// first get transform type
		ret = sscanf(optarg, "%4[^:]", rt);
		assert(1 == ret);

		if (strcmp(rt, "W") == 0) {

			regs[r].xform = L1WAV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);

		} else 
		if (strcmp(rt, "L") == 0) {

			regs[r].xform = LLR;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);

		} else 
		if (strcmp(rt, "Q") == 0) {

			regs[r].xform = L2IMG;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;

		} else 
		if (strcmp(rt, "S") == 0) {

			regs[r].xform = POS;
			int ret = sscanf(optarg, "%*[^:]:%f", &regs[r].lambda);
			assert(1 == ret);
			regs[r].xflags = 0u;
			regs[r].jflags = 0u;

		} else 
		if (strcmp(rt, "h") == 0) {

			help_reg_moba();
			exit(0);

		} else {

			error(" > Unrecognized regularization type: \"%s\"\n", rt);
		}

		p->r++;
		break;
	}

	return false;
}


static void opt_reg_meco_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int model)
{
	long maps_dims[N];
	md_select_dims(N, ~COIL_FLAG, maps_dims, dims);

	long maps_size = md_calc_size(N, maps_dims);

	long sens_dims[N];
	md_select_dims(N, ~COEFF_FLAG, sens_dims, dims);

	long sens_size = md_calc_size(N, sens_dims);

	long js_dim = COEFF_DIM; // joint spatial dim
	long nr_coeff = maps_dims[js_dim];

	long jt_dim = TIME_DIM;  // joint temporal dim
	long nr_time = maps_dims[jt_dim];
	UNUSED(nr_time);


	// flatten number of maps and coils
	long x_dims[N];
	md_select_dims(N, ~(COIL_FLAG|TE_FLAG|COEFF_FLAG), x_dims, maps_dims);
	x_dims[js_dim] = nr_coeff + sens_dims[COIL_DIM];

	struct reg_s* regs = ropts->regs;
	int nr_penalties = ropts->r;

	debug_printf(DP_INFO, " >> in total %1d regularizations:\n", nr_penalties);

	for (int nr = 0; nr < nr_penalties; nr++) {

		switch (regs[nr].xform) {

		case L1WAV:

			debug_printf(DP_INFO, "  > l1-wavelet regularization\n");

			prox_ops[nr] = create_stack_spatial_thresh_prox(N, x_dims, js_dim, L1WAV, regs[nr].lambda, model);
			trafos[nr] = linop_identity_create(DIMS, x_dims);

			break;

		case LLR:

			debug_printf(DP_INFO, "  > lowrank regularization\n");

			prox_ops[nr] = create_stack_spatial_thresh_prox(N, x_dims, js_dim, LLR, regs[nr].lambda, model);
			trafos[nr] = linop_identity_create(DIMS, x_dims);

			break;

		case L2IMG:

			debug_printf(DP_INFO, "  > l2 regularization\n");

			prox_ops[nr] = prox_l2norm_create(N, x_dims, regs[nr].lambda);
			trafos[nr] = linop_identity_create(N, x_dims);

			break;

		case POS:
		{
			debug_printf(DP_INFO, "  > non-negative constraint with lambda %f\n", regs[nr].lambda);

			auto prox_maps = moba_nonneg_prox_create(N, maps_dims, js_dim, set_R2S_flag(model), regs[nr].lambda);
			auto prox_sens = moba_sens_prox_create(N, sens_dims);

			prox_ops[nr] = stack_flatten_prox(prox_maps, prox_sens);

			trafos[nr] = linop_identity_create(1, MD_DIMS(maps_size + sens_size));

			break;
		}

		default:

			prox_ops[nr] = NULL;
			trafos[nr] = NULL;

			break;
		}
	}
}


void opt_reg_moba_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], unsigned int model)
{
	switch (model) {

	case MECO_WF:
	case MECO_WFR2S:
	case MECO_WF2R2S:
	case MECO_R2S:
	case MECO_PHASEDIFF:

		opt_reg_meco_configure(N, dims, ropts, prox_ops, trafos, model);

		break;
	}
}

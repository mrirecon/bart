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

struct optreg_conf optreg_defaults = {

	.moba_model = MECO_WFR2S,
	.weight_fB0_type = MECO_SOBOLEV,
};


static const struct operator_p_s* create_wav_prox(const long img_dims[DIMS], unsigned int x_flags, unsigned int jt_flag, float lambda)
{
	bool randshift = true;
	long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
	unsigned int wflags = 0;

	for (unsigned int i = 0; i < DIMS; i++) {

		if ((1 < img_dims[i]) && MD_IS_SET(x_flags, i)) {

			wflags = MD_SET(wflags, i);
			minsize[i] = MIN(img_dims[i], DIMS);
		}
	}

	return prox_wavelet_thresh_create(DIMS, img_dims, wflags, jt_flag, minsize, lambda, randshift);
}


static const struct operator_p_s* ops_p_stack_higher_dims(unsigned int N, const long maps_dims[N], unsigned int coeff_dim, long higher_flag, const struct operator_p_s* src)
{
	const struct operator_p_s* tmp = operator_p_ref(src);
	const struct operator_p_s* dst = operator_p_ref(src);

	for (long d = coeff_dim + 1; d < N; d++) {

		if (MD_IS_SET(higher_flag, d)) {

			for (long p = 1; p < maps_dims[d]; p++)
				dst = operator_p_stack_FF(d, d, dst, operator_p_ref(tmp));

			operator_p_free(tmp);

			tmp = operator_p_ref(dst);
		}
	}

	operator_p_free(tmp);

	return dst;
}

static const struct operator_p_s* moba_joint_wavthresh_prox_create(unsigned int N, const long maps_dims[N], long coeff_dim, long x_flags, long jflag, float lambda, long nr_joint_maps)
{
	// higher dimensions
	long higher_flag = 0;

	for (long d = coeff_dim + 1; d < N; d++)
		if (1 < maps_dims[d])
			higher_flag = MD_SET(higher_flag, d);

	long maps_j_dims[N];
	md_select_dims(N, ~(MD_BIT(coeff_dim)|higher_flag), maps_j_dims, maps_dims);
	maps_j_dims[coeff_dim] = nr_joint_maps;

	auto prox_j = create_wav_prox(maps_j_dims, x_flags, jflag, lambda);

	if (nr_joint_maps < maps_dims[coeff_dim]) {

		long maps_z_dims[N];
		md_select_dims(N, ~(MD_BIT(coeff_dim)|higher_flag), maps_z_dims, maps_dims);
		maps_z_dims[coeff_dim] = maps_dims[coeff_dim] - nr_joint_maps;

		auto prox_z = prox_zero_create(N, maps_z_dims);

		prox_j = operator_p_stack_FF(coeff_dim, coeff_dim, prox_j, prox_z);
	}

	// stack higher dimensions
	auto prox_s = ops_p_stack_higher_dims(N, maps_dims, coeff_dim, higher_flag, prox_j);

	operator_p_free(prox_j);

	return prox_s;
}


const struct operator_p_s* moba_nonneg_prox_create(unsigned int N, const long maps_dims[N], unsigned int coeff_dim, long nonneg_flag, float lambda)
{
	// higher dimensions
	long higher_flag = 0;

	for (long d = coeff_dim + 1; d < N; d++) {

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

		if (NULL == prox_j) {

			prox_j = operator_p_ref(p3);

		} else {

			auto tmp = operator_p_stack(coeff_dim, coeff_dim, prox_j, p3);
			operator_p_free(prox_j);
			prox_j = tmp;
		}
	}

	operator_p_free(p1);
	operator_p_free(p2);
//	operator_p_free(p3);

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
			"-R W:A:B:C\tl1-wavelet\n"
			"-R Q:C\tl2 regularization\n"
			"-R S:C\tnon-negative constraint\n"
			"-R T:A:B:C\ttotal variation\n");
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
		if (strcmp(rt, "T") == 0) {

			regs[r].xform = TV;
			int ret = sscanf(optarg, "%*[^:]:%d:%d:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].lambda);
			assert(3 == ret);

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


static void opt_reg_meco_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], struct optreg_conf* optreg_conf)
{
	long maps_dims[N];
	md_select_dims(N, ~COIL_FLAG, maps_dims, dims);

	long maps_size = md_calc_size(N, maps_dims);

	long sens_dims[N];
	md_select_dims(N, ~COEFF_FLAG, sens_dims, dims);

	long sens_size = md_calc_size(N, sens_dims);

	long x_size = maps_size + sens_size;


	// set number of coefficients for joint regularization
	long nr_joint_coeff = set_num_of_coeff(optreg_conf->moba_model);

	if (MECO_SOBOLEV == optreg_conf->weight_fB0_type)
		nr_joint_coeff -= 1;

	// set the flag for the position of the coefficient 
	// which needs non-negativity constraint
	long nonneg_flag = set_R2S_flag(optreg_conf->moba_model);


	struct reg_s* regs = ropts->regs;
	int nr_penalties = ropts->r;

	debug_printf(DP_INFO, " >> in total %1d regularizations:\n", nr_penalties);

	for (int nr = 0; nr < nr_penalties; nr++) {

		switch (regs[nr].xform) {

		case L1WAV:
		{
			debug_printf(DP_INFO, "  > l1-wavelet regularization with parameters %d:%d:%.3f\n", regs[nr].xflags, regs[nr].jflags, regs[nr].lambda);

			auto prox_maps = moba_joint_wavthresh_prox_create(N, maps_dims, COEFF_DIM, regs[nr].xflags, regs[nr].jflags, regs[nr].lambda, nr_joint_coeff);

			auto prox_sens = moba_sens_prox_create(N, sens_dims);

			prox_ops[nr] = stack_flatten_prox(prox_maps, prox_sens);

			trafos[nr] = linop_identity_create(1, MD_DIMS(x_size));

			break;
		}

		case L2IMG:
		{
			debug_printf(DP_INFO, "  > l2 regularization\n");

			prox_ops[nr] = prox_l2norm_create(1, MD_DIMS(x_size), regs[nr].lambda);
			trafos[nr] = linop_identity_create(1, MD_DIMS(x_size));

			break;
		}

		case POS:
		{
			debug_printf(DP_INFO, "  > non-negative constraint with lambda %f\n", regs[nr].lambda);

			auto prox_maps = moba_nonneg_prox_create(N, maps_dims, COEFF_DIM, nonneg_flag, regs[nr].lambda);

			auto prox_sens = moba_sens_prox_create(N, sens_dims);

			prox_ops[nr] = stack_flatten_prox(prox_maps, prox_sens);

			trafos[nr] = linop_identity_create(1, MD_DIMS(x_size));

			break;
		}

		case TV: // temporal dimension
		{
			debug_printf(DP_INFO, "  > TV regularization with parameters %d:%d:%.3f\n", regs[nr].xflags, regs[nr].jflags, regs[nr].lambda);

			auto lo_extract_maps = linop_extract_create(1, MD_DIMS(0), MD_DIMS(maps_size), MD_DIMS(x_size));
			lo_extract_maps = linop_reshape_out_F(lo_extract_maps, N, maps_dims);

			auto lo_grad_maps = linop_grad_create(N, maps_dims, N, regs[nr].xflags);

			trafos[nr] = linop_chain_FF(lo_extract_maps, lo_grad_maps);

			prox_ops[nr] = prox_thresh_create(N + 1,
					linop_codomain(trafos[nr])->dims,
					regs[nr].lambda, regs[nr].jflags | MD_BIT(N));

			break;
		}

		default:

			prox_ops[nr] = NULL;
			trafos[nr] = NULL;

			break;
		}
	}
}


void opt_reg_moba_configure(unsigned int N, const long dims[N], struct opt_reg_s* ropts, const struct operator_p_s* prox_ops[NUM_REGS], const struct linop_s* trafos[NUM_REGS], struct optreg_conf* optreg_conf)
{
	switch (optreg_conf->moba_model) {

	case MECO_WF:
	case MECO_WFR2S:
	case MECO_WF2R2S:
	case MECO_R2S:
	case MECO_PHASEDIFF:

		opt_reg_meco_configure(N, dims, ropts, prox_ops, trafos, optreg_conf);

		break;
	}
}

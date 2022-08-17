/* Copyright 2020. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "num/gpuops.h"
#include "num/fft.h"
#include "num/filter.h"
#include "num/flpmath.h"
#include "num/multind.h"

#include "simu/signals.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"

#include "noir/model.h"
#include "noir/utils.h"

#include "meco.h"


struct meco_s {

	INTERFACE(nlop_data_t);

	int N;
	long model;

	bool real_pd;

	const long* y_dims;
	const long* x_dims;
	const long* der_dims;
	const long* map_dims;
	const long* TE_dims;

	const long* y_strs;
	const long* x_strs;
	const long* der_strs;
	const long* map_strs;
	const long* TE_strs;

	// Parameter maps
	complex float* der_x;
	complex float* TE;
	complex float* cshift;
	complex float* scaling; // length = number of maps
	complex float* weights;

	const struct linop_s* linop_fB0;
	unsigned int weight_fB0_type;
};

DEF_TYPEID(meco_s);


int get_num_of_coeff(enum meco_model sel_model)
{
	int ncoeff = 0;

	switch (sel_model) {
	case MECO_PI:		assert(0);
	case MECO_WF: 		ncoeff = 3; break;
	case MECO_WFR2S:	ncoeff = 4; break;
	case MECO_WF2R2S:	ncoeff = 5; break;
	case MECO_R2S:		ncoeff = 3; break;
	case MECO_PHASEDIFF:	ncoeff = 2; break;
	}

	return ncoeff;
}

long get_PD_flag(enum meco_model sel_model)
{
	long PD_flag = 0;

	switch (sel_model) {

	case MECO_PI:

		assert(0);

	case MECO_WF:

		PD_flag = MD_SET(PD_flag, 0);
		PD_flag = MD_SET(PD_flag, 1);
		break;

	case MECO_WFR2S:

		PD_flag = MD_SET(PD_flag, 0);
		PD_flag = MD_SET(PD_flag, 1);
		break;

	case MECO_WF2R2S:

		PD_flag = MD_SET(PD_flag, 0);
		PD_flag = MD_SET(PD_flag, 2);
		break;

	case MECO_R2S:

		PD_flag = MD_SET(PD_flag, 0);
		break;

	case MECO_PHASEDIFF:

		PD_flag = MD_SET(PD_flag, 0);
		break;
	}

	return PD_flag;
}

long get_R2S_flag(enum meco_model sel_model)
{
	long R2S_flag = 0;

	switch (sel_model) {

	case MECO_PI:

		assert(0);

	case MECO_WF:

		break;

	case MECO_WFR2S:

		R2S_flag = MD_SET(R2S_flag, 2);
		break;

	case MECO_WF2R2S:

		R2S_flag = MD_SET(R2S_flag, 1);
		R2S_flag = MD_SET(R2S_flag, 3);
		break;

	case MECO_R2S:

		R2S_flag = MD_SET(R2S_flag, 1);
		break;

	case MECO_PHASEDIFF:

		break;
	}

	return R2S_flag;
}

long get_fB0_flag(enum meco_model sel_model)
{
	// the last parameter is fB0
	long fB0_flag = 0;

	fB0_flag = MD_SET(fB0_flag, get_num_of_coeff(sel_model) - 1);

	return fB0_flag;
}

void meco_calc_fat_modu(int N, const long dims[N], const complex float TE[dims[TE_DIM]], complex float dst[dims[TE_DIM]], enum fat_spec fat_spec)
{
	md_clear(N, dims, dst, CFL_SIZE);

	for (int eind = 0; eind < dims[TE_DIM]; eind++) {

		assert(0. == cimagf(TE[eind]));

		dst[eind] = calc_fat_modulation(3.0, crealf(TE[eind]) * 1.E-3, fat_spec);
	}
}



static void meco_calc_weights(const nlop_data_t* _data, const int N, const long dims[N], float wgh_fB0)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	enum meco_weights_fB0 weights_type = (0. == wgh_fB0) ? MECO_IDENTITY : MECO_SOBOLEV;

	switch (weights_type) {

	case MECO_IDENTITY:

		debug_printf(DP_DEBUG2, " identity weight on fB0\n");

		md_zfill(N, dims, data->weights, 1.);

		data->linop_fB0 = linop_cdiag_create(N, data->map_dims, FFT_FLAGS, data->weights);

		data->weight_fB0_type = MECO_IDENTITY;

		break;

	case MECO_SOBOLEV:

		debug_printf(DP_DEBUG2, " sobolev weight on fB0\n");

		noir_calc_weights(wgh_fB0, 32., dims, data->weights);

		auto linop_wghts = linop_cdiag_create(N, data->map_dims, FFT_FLAGS, data->weights);
		auto linop_ifftc = linop_ifftc_create(N, data->map_dims, FFT_FLAGS);

		data->linop_fB0 = linop_chain_FF(linop_wghts, linop_ifftc);

		data->weight_fB0_type = MECO_SOBOLEV;

		break;

	default:

		assert(0);
		break;
	}
}

const complex float* meco_get_scaling(struct nlop_s* op)
{
	const nlop_data_t* _data = nlop_get_data(op);
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	return data->scaling;
}

const struct linop_s* meco_get_fB0_trafo(struct nlop_s* op)
{
	const nlop_data_t* _data = nlop_get_data(op);
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	return data->linop_fB0;
}

void meco_forw_fB0(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void meco_back_fB0(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}

unsigned int meco_get_weight_fB0_type(struct nlop_s* op)
{
	const nlop_data_t* _data = nlop_get_data(op);
	struct meco_s* data = CAST_DOWN(meco_s, _data);
	return data->weight_fB0_type;
}

// ************************************************************* //
//  Model: (W + F cshift) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wf(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;

	enum { PIND_W = 0, PIND_F = 1, PIND_FB0 = 2 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// F
	x_pos[COEFF_DIM] = PIND_F;

	complex float* F = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, F, data->x_dims, src, CFL_SIZE);

	// dst = F .* cshift
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, F, data->TE_strs, data->cshift);
	md_zsmul(data->N, data->y_dims, dst, dst, data->scaling[PIND_F]);


	// W
	x_pos[COEFF_DIM] = PIND_W;

	complex float* W = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, W, data->x_dims, src, CFL_SIZE);

	// dst = W + F .* cshift
	md_zadd2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, dst, data->map_strs, W);


	// fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	meco_forw_fB0(data->linop_fB0, fB0, fB0);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, fB0, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_exp, tmp_exp, 2.i * M_PI * data->scaling[PIND_FB0]);

	// tmp_exp = exp(1i*2*pi * fB0 .* TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);

	// dst = dst .* tmp_exp
	md_zmul(data->N, data->y_dims, dst, dst, tmp_exp);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	x_pos[COEFF_DIM] = PIND_W;
	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);

	// der_F
	x_pos[COEFF_DIM] = PIND_F;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_exp, data->TE_strs, data->cshift);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, data->scaling[PIND_F]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, I*2.*M_PI * data->scaling[PIND_FB0]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(W);
	md_free(F);
	md_free(fB0);
}


// ************************************************************* //
//  Model: (W + F cshift) .* exp(- R2s TE) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wfr2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_W = 0, PIND_F = 1, PIND_R2S = 2, PIND_FB0 = 3 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);


	// =============================== //
	//  forward operator
	// =============================== //

	// F
	x_pos[COEFF_DIM] = PIND_F;

	complex float* F = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, F, data->x_dims, src, CFL_SIZE);

	// dst = F .* cshift
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, F, data->TE_strs, data->cshift);
	md_zsmul(data->N, data->y_dims, dst, dst, data->scaling[PIND_F]);


	// W
	x_pos[COEFF_DIM] = PIND_W;

	complex float* W = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, W, data->x_dims, src, CFL_SIZE);

	// dst = W + F .* cshift
	md_zadd2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, dst, data->map_strs, W);


	// R2s and fB0
	x_pos[COEFF_DIM] = PIND_R2S;

	complex float* R2s = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, R2s, data->x_dims, src, CFL_SIZE);

	md_zsmul(data->N, data->map_dims, R2s, R2s, -1. * data->scaling[PIND_R2S]);


	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	meco_forw_fB0(data->linop_fB0, fB0, fB0);

	md_zaxpy2(data->N, data->map_dims, data->map_strs, R2s, 2.i * M_PI * data->scaling[PIND_FB0], data->map_strs, fB0);
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, R2s, data->TE_strs, data->TE);

	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);

	// dst = dst .* tmp_exp
	md_zmul(data->N, data->y_dims, dst, dst, tmp_exp);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	x_pos[COEFF_DIM] = PIND_W;

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);

	// der_F
	x_pos[COEFF_DIM] = PIND_F;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_exp, data->TE_strs, data->cshift);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, data->scaling[PIND_F]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_R2s
	x_pos[COEFF_DIM] = PIND_R2S;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1. * data->scaling[PIND_R2S]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI * data->scaling[PIND_FB0]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(W);
	md_free(F);
	md_free(R2s);
	md_free(fB0);
}


// ************************************************************* //
//  Model: (W exp(- R2s_W TE) + F cshift exp(- R2s_F TE)) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_wf2r2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_W = 0, PIND_R2SW = 1, PIND_F = 2, PIND_R2SF = 3, PIND_FB0 = 4 };

	complex float* tmp_exp_R2sW = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_exp_R2sF = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_exp_fB0  = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco      = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// W
	x_pos[COEFF_DIM] = PIND_W;

	complex float* W = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, W, data->x_dims, src, CFL_SIZE);


	// R2sW
	x_pos[COEFF_DIM] = PIND_R2SW;

	complex float* R2sW = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, R2sW, data->x_dims, src, CFL_SIZE);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp_R2sW, data->map_strs, R2sW, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_exp_R2sW, tmp_exp_R2sW, -1. * data->scaling[PIND_R2SW]);


	// F
	x_pos[COEFF_DIM] = PIND_F;

	complex float* F = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, F, data->x_dims, src, CFL_SIZE);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, F, data->TE_strs, data->cshift);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, data->scaling[PIND_F]);


	// R2sF
	x_pos[COEFF_DIM] = PIND_R2SF;

	complex float* R2sF = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	md_copy_block(data->N, x_pos, data->map_dims, R2sF, data->x_dims, src, CFL_SIZE);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp_R2sF, data->map_strs, R2sF, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_exp_R2sF, tmp_exp_R2sF, -1. * data->scaling[PIND_R2SF]);


	// fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	meco_forw_fB0(data->linop_fB0, fB0, fB0);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp_fB0, data->map_strs, fB0, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_exp_fB0, tmp_exp_fB0, 2.i * M_PI * data->scaling[PIND_FB0]);

	// tmp_exp_R2sW = exp(- R2sW TE)
	md_zexp(data->N, data->y_dims, tmp_exp_R2sW, tmp_exp_R2sW);

	// tmp_exp_R2sF = exp(- R2sF TE)
	md_zexp(data->N, data->y_dims, tmp_exp_R2sF, tmp_exp_R2sF);

	// tmp_exp_fB0 = exp(i 2\pi fB0 TE)
	md_zexp(data->N, data->y_dims, tmp_exp_fB0, tmp_exp_fB0);

	// tmp_eco = W exp(- R2s_W TE) + F cshift exp(- R2s_F TE)
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_eco, tmp_exp_R2sF);
	md_zfmac2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, W, data->y_strs, tmp_exp_R2sW);

	// dst = tmp_eco .* tmp_exp_fB0
	md_zmul(data->N, data->y_dims, dst, tmp_eco, tmp_exp_fB0);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_W
	x_pos[COEFF_DIM] = PIND_W;
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_exp_fB0, tmp_exp_R2sW);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_R2sW
	x_pos[COEFF_DIM] = PIND_R2SW;
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, W, data->y_strs, tmp_eco);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1. * data->scaling[PIND_R2SW]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_F
	x_pos[COEFF_DIM] = PIND_F;
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_exp_fB0, tmp_exp_R2sF);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, data->cshift);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, data->scaling[PIND_F]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_R2sF
	x_pos[COEFF_DIM] = PIND_R2SF;
	md_zmul(data->N, data->y_dims, tmp_eco, tmp_exp_fB0, tmp_exp_R2sF);
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->map_strs, F, data->y_strs, tmp_eco);


	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, data->cshift);
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, tmp_eco, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1. * data->scaling[PIND_R2SF]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI * data->scaling[PIND_FB0]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp_fB0);
	md_free(tmp_exp_R2sW);
	md_free(tmp_exp_R2sF);
	md_free(tmp_eco);
	md_free(W);
	md_free(R2sW);
	md_free(F);
	md_free(R2sF);
	md_free(fB0);
}


// ************************************************************* //
//  Model: rho .* exp(- R2s TE) .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_r2s(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_RHO = 0, PIND_R2S = 1, PIND_FB0 = 2 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// R2s and fB0
	x_pos[COEFF_DIM] = PIND_R2S;

	complex float* R2s = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, R2s, data->x_dims, src, CFL_SIZE);

	md_zsmul(data->N, data->map_dims, R2s, R2s, -1. * data->scaling[PIND_R2S]);


	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	meco_forw_fB0(data->linop_fB0, fB0, fB0);

	md_zaxpy2(data->N, data->map_dims, data->map_strs, R2s, 2.i * M_PI * data->scaling[PIND_FB0], data->map_strs, fB0);

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, R2s, data->TE_strs, data->TE);

	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);


	// rho
	x_pos[COEFF_DIM] = PIND_RHO;

	complex float* rho = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, rho, data->x_dims, src, CFL_SIZE);

	// dst = tmp_exp .* rho
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, tmp_exp, data->map_strs, rho);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_rho
	x_pos[COEFF_DIM] = PIND_RHO;

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);


	// der_R2s
	x_pos[COEFF_DIM] = PIND_R2S;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, -1. * data->scaling[PIND_R2S]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);


	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI * data->scaling[PIND_FB0]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(rho);
	md_free(R2s);
	md_free(fB0);
}


// ************************************************************* //
//  Model: rho .* exp(i 2\pi fB0 TE)
// ************************************************************* //
static void meco_fun_phasediff(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	enum { PIND_RHO = 0, PIND_FB0 = 1 };

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);
	complex float* tmp_eco = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	// =============================== //
	//  forward operator
	// =============================== //

	// R2s and fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	complex float* fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, fB0, data->x_dims, src, CFL_SIZE);

	meco_forw_fB0(data->linop_fB0, fB0, fB0);

	md_zaxpy2(data->N, data->map_dims, data->map_strs, fB0, 2.i * M_PI * data->scaling[PIND_FB0], data->map_strs, fB0);
	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_exp, data->map_strs, fB0, data->TE_strs, data->TE);


	// tmp_exp = exp(z TE)
	md_zexp(data->N, data->y_dims, tmp_exp, tmp_exp);


	// rho
	x_pos[COEFF_DIM] = PIND_RHO;

	complex float* rho = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	md_copy_block(data->N, x_pos, data->map_dims, rho, data->x_dims, src, CFL_SIZE);

	// dst = tmp_exp .* rho
	md_zmul2(data->N, data->y_dims, data->y_strs, dst, data->y_strs, tmp_exp, data->map_strs, rho);


	// =============================== //
	//  partial derivative operator
	// =============================== //
	// der_rho
	x_pos[COEFF_DIM] = PIND_RHO;

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_exp, CFL_SIZE);

	// der_fB0
	x_pos[COEFF_DIM] = PIND_FB0;

	md_zmul2(data->N, data->y_dims, data->y_strs, tmp_eco, data->y_strs, dst, data->TE_strs, data->TE);
	md_zsmul(data->N, data->y_dims, tmp_eco, tmp_eco, 2.i * M_PI * data->scaling[PIND_FB0]);

	md_copy_block(data->N, x_pos, data->der_dims, data->der_x, data->y_dims, tmp_eco, CFL_SIZE);

	md_free(tmp_exp);
	md_free(tmp_eco);
	md_free(rho);
	md_free(fB0);
}


static void meco_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	md_clear(data->N, data->y_dims, dst, CFL_SIZE);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {

		x_pos[COEFF_DIM] = pind;

		md_copy_block(data->N, x_pos, data->map_dims, tmp_map, data->x_dims, src, CFL_SIZE);
		md_copy_block(data->N, x_pos, data->y_dims, tmp_exp, data->der_dims, data->der_x, CFL_SIZE);

		if (pind == data->x_dims[COEFF_DIM] - 1)
			meco_forw_fB0(data->linop_fB0, tmp_map, tmp_map);

		md_zfmac2(data->N, data->y_dims, data->y_strs, dst, data->map_strs, tmp_map, data->y_strs, tmp_exp);
	}

	md_free(tmp_map);
	md_free(tmp_exp);
}

static void meco_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	struct meco_s* data = CAST_DOWN(meco_s, _data);

	long x_pos[data->N];

	for (int i = 0; i < data->N; i++)
		x_pos[i] = 0;


	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->y_dims, CFL_SIZE, dst);

	md_clear(data->N, data->x_dims, dst, CFL_SIZE);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {

		x_pos[COEFF_DIM] = pind;

		md_copy_block(data->N, x_pos, data->map_dims, tmp_map, data->x_dims, dst, CFL_SIZE);
		md_copy_block(data->N, x_pos, data->y_dims, tmp_exp, data->der_dims, data->der_x, CFL_SIZE);

		md_zfmacc2(data->N, data->y_dims, data->map_strs, tmp_map, data->y_strs, src, data->y_strs, tmp_exp);

		md_copy_block(data->N, x_pos, data->x_dims, dst, data->map_dims, tmp_map, CFL_SIZE);
	}


	// real constraint
	long  PD_flag = get_PD_flag(data->model);
	long R2S_flag = get_R2S_flag(data->model);
	long fB0_flag = get_fB0_flag(data->model);

	for (long pind = 0; pind < data->x_dims[COEFF_DIM]; pind++) {

		if (  (MD_IS_SET(PD_flag, pind) && data->real_pd)
		    || MD_IS_SET(R2S_flag, pind)
		    || MD_IS_SET(fB0_flag, pind)) {

			x_pos[COEFF_DIM] = pind;

			md_copy_block(data->N, x_pos, data->map_dims, tmp_map, data->x_dims, dst, CFL_SIZE);
#if 1
			md_zreal(data->N, data->map_dims, tmp_map, tmp_map);
#endif
			if (MD_IS_SET(fB0_flag, pind))
				meco_back_fB0(data->linop_fB0, tmp_map, tmp_map);

			md_copy_block(data->N, x_pos, data->x_dims, dst, data->map_dims, tmp_map, CFL_SIZE);
		}
	}

	md_free(tmp_map);
	md_free(tmp_exp);
}

static void meco_del(const nlop_data_t* _data)
{
	struct meco_s* data = CAST_DOWN(meco_s, _data);

	md_free(data->TE);
	md_free(data->cshift);
	md_free(data->scaling);
	md_free(data->weights);

	md_free(data->der_x);

	xfree(data->y_dims);
	xfree(data->x_dims);
	xfree(data->der_dims);
	xfree(data->map_dims);
	xfree(data->TE_dims);

	xfree(data->y_strs);
	xfree(data->x_strs);
	xfree(data->der_strs);
	xfree(data->map_strs);
	xfree(data->TE_strs);

	linop_free(data->linop_fB0);

	xfree(data);
}


struct nlop_s* nlop_meco_create(const int N, const long y_dims[N], const long x_dims[N], const complex float* TE, enum meco_model sel_model, bool real_pd, enum fat_spec fat_spec, const float* scale_fB0, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

	PTR_ALLOC(struct meco_s, data);
	SET_TYPEID(meco_s, data);


	PTR_ALLOC(long[N], nydims);
	md_copy_dims(N, *nydims, y_dims);
	data->y_dims = *PTR_PASS(nydims);

	assert(x_dims[COEFF_DIM] == get_num_of_coeff(sel_model));
	data->model = sel_model;

	PTR_ALLOC(long[N], nxdims);
	md_copy_dims(N, *nxdims, x_dims);
	data->x_dims = *PTR_PASS(nxdims);

	PTR_ALLOC(long[N], nderdims);
	md_merge_dims(N, *nderdims, y_dims, x_dims);
	data->der_dims = *PTR_PASS(nderdims);

	long map_dims[N];
	md_select_dims(N, ~COEFF_FLAG, map_dims, x_dims);
	PTR_ALLOC(long[N], n1dims);
	md_copy_dims(N, *n1dims, map_dims);
	data->map_dims = *PTR_PASS(n1dims);

	long TE_dims[N];
	md_select_dims(N, TE_FLAG, TE_dims, y_dims);
	PTR_ALLOC(long[N], ntedims);
	md_copy_dims(N, *ntedims, TE_dims);
	data->TE_dims = *PTR_PASS(ntedims);

	long scaling_dims[N];
	md_select_dims(N, COEFF_FLAG, scaling_dims, x_dims);


	PTR_ALLOC(long[N], nystr);
	md_calc_strides(N, *nystr, y_dims, CFL_SIZE);
	data->y_strs = *PTR_PASS(nystr);

	PTR_ALLOC(long[N], nxstr);
	md_calc_strides(N, *nxstr, x_dims, CFL_SIZE);
	data->x_strs = *PTR_PASS(nxstr);

	PTR_ALLOC(long[N], nderstr);
	md_calc_strides(N, *nderstr, data->der_dims, CFL_SIZE);
	data->der_strs = *PTR_PASS(nderstr);

	PTR_ALLOC(long[N], n1str);
	md_calc_strides(N, *n1str, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(n1str);

	PTR_ALLOC(long[N], ntestr);
	md_calc_strides(N, *ntestr, TE_dims, CFL_SIZE);
	data->TE_strs = *PTR_PASS(ntestr);

	data->N = N;
	data->der_x = my_alloc(N, data->der_dims, CFL_SIZE);

	// echo times
	data->TE = my_alloc(N, TE_dims, CFL_SIZE);
	md_copy(N, TE_dims, data->TE, TE, CFL_SIZE);


	// calculate cshift
	complex float* cshift = md_alloc(N, TE_dims, CFL_SIZE);

	meco_calc_fat_modu(N, TE_dims, TE, cshift, fat_spec);

	data->cshift = my_alloc(N, TE_dims, CFL_SIZE);
	md_copy(N, TE_dims, data->cshift, cshift, CFL_SIZE);

	md_free(cshift);

	// weight on fB0
	long w_dims[N];
	md_select_dims(N, FFT_FLAGS, w_dims, data->x_dims);

	data->weights = md_alloc(N, w_dims, CFL_SIZE);

	meco_calc_weights(CAST_UP(data), N, w_dims, scale_fB0[0]);

	// scaling
	data->scaling = md_alloc(N, scaling_dims, CFL_SIZE);

	for (int pind = 0; pind < x_dims[COEFF_DIM]; pind++)
		data->scaling[pind] = 1.0;

	long fB0_ind = x_dims[COEFF_DIM] - 1;
	data->scaling[fB0_ind] = scale_fB0[1];

	nlop_fun_t meco_funs[] = {

		[MECO_WF] = meco_fun_wf,
		[MECO_WFR2S] = meco_fun_wfr2s,
		[MECO_WF2R2S] = meco_fun_wf2r2s,
		[MECO_R2S] = meco_fun_r2s,
		[MECO_PHASEDIFF] = meco_fun_phasediff,
	};

	data->real_pd = real_pd;

	return nlop_create(N, y_dims, N, x_dims, CAST_UP(PTR_PASS(data)), meco_funs[sel_model], meco_der, meco_adj, NULL, NULL, meco_del);
}

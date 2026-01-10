/* Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Nick Scholand, Martin Uecker
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"

#include "simu/signals.h"
#include "num/filter.h"

#include "nlops/nlop.h"
#include "linops/linop.h"
#include "linops/someops.h"
#include "noir/utils.h"


#include "ir_meco.h"

#include "meco.h"

int ir_meco_get_num_of_coeff(enum meco_model sel_model)
{
	int ncoeff = 0;

	switch (sel_model) {

	case IR_MECO_WF_fB0:		ncoeff = 3; break; // meco, water, fat, fB0
	case IR_MECO_WF_R2S:		ncoeff = 4; break; // meco, water, fat, R2*, fB0
	case IR_MECO_T1_R2S:		ncoeff = 5; break; // ir + meco, water T1, R2*, fB0
	case IR_MECO_W_T1_F_T1_R2S:	ncoeff = 8; break; // ir + meco, water T1, fat T1, R2*, fB0
	default: error("invalid model");
	}

	return ncoeff;
}

struct ir_meco_s {

	nlop_data_t super;

	int N;

	const long* map_dims;
	const long* TI_dims;
	const long* TE_dims;
	const long* in_dims;

	const long* out_dims; // for IR + meco
	const long* out2_dims; // only for IR
	const long* out3_dims; // only for meco

	const long* map_strs;
	const long* TI_strs;
	const long* TE_strs;
	const long* in_strs;
	const long* out_strs; // for IR + meco
	const long* out2_strs; // only for IR
	const long* out3_strs; // only for meco


	// Parameter maps
	// water
	complex float* Ms_w;
	complex float* M0_w;
	complex float* R1s_w;

	// fat
	complex float* Ms_f;
	complex float* M0_f;
	complex float* R1s_f;

	// off-resonance
	complex float* fB0;

	// R2star
	complex float* R2s;

	complex float* tmp_map;

	complex float* dMs_w;
	complex float* dM0_w;
	complex float* dR1s_w;

	complex float* dMs_f;
	complex float* dM0_f;
	complex float* dR1s_f;

	complex float* dR2s;
	complex float* dfB0;

	struct multiplace_array_s* TI;
	struct multiplace_array_s* TE;
	struct multiplace_array_s* cshift;

	float scaling_M0_w;
	float scaling_R1s_w;
	float scaling_F;
	float scaling_M0_f;
	float scaling_R1s_f;
	float scaling_R2s;
	float scaling_fB0;

	const struct linop_s* linop_fB0;

};

DEF_TYPEID(ir_meco_s);


void ir_meco_calc_fat_modu(int N, const long dims[N], const complex float TE[dims[CSHIFT_DIM]], complex float dst[dims[CSHIFT_DIM]], enum fat_spec fat_spec)
{
	md_clear(N, dims, dst, CFL_SIZE);

	for (int eind = 0; eind < dims[CSHIFT_DIM]; eind++) {

		assert(0. == cimagf(TE[eind]));

		dst[eind] = calc_fat_modulation(3.0, crealf(TE[eind]) * 1.E-3, fat_spec); // FIXME: TE in SI units instead ms
	}
}


const struct linop_s* ir_meco_get_fB0_trafo(struct nlop_s* op)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, nlop_get_data(op));

	return data->linop_fB0;
}

void ir_meco_forw_fB0(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void ir_meco_back_fB0(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}

static void ir_meco_init(struct ir_meco_s* data, const void* arg)
{
	if (NULL != data->Ms_w)
		return;

	// Default model
	data->Ms_w = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
	data->fB0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
	data->tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);

	data->dMs_w = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
	data->dfB0 = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);

	if (3 != data->in_dims[COEFF_DIM]) {

		data->R2s = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
		data->dR2s = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
	}

	if (5 != data->in_dims[COEFF_DIM]) {

		data->Ms_f = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
		data->dMs_f = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
	}

	if (8 == data->in_dims[COEFF_DIM]) {

		data->M0_f = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
		data->R1s_f = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);

		data->dM0_f = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
		data->dR1s_f = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
	}

	if ((5 == data->in_dims[COEFF_DIM]) || (8 == data->in_dims[COEFF_DIM])) {

		data->M0_w = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
		data->R1s_w = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);

		data->dM0_w = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
		data->dR1s_w = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
	}
}


// Calculate Model:
// Water = (Ms_w - (Ms_w + M0_w) * exp(-TI_k.*R1s_w))  // only for IR
// Fat = (Ms_f - (Ms_f + M0_f) * exp(-TI_k.*R1s_f)) // only for meco
// Water-fat model: (Water + Fat * z_m) * exp(i * 2pi * f_B0 * TE_m) // for IR meco

// Water-fat, R2* model: (Water + Fat * z_m) * exp(i * 2pi * f_B0 * TE_m) * exp(i * 2pi * f_B0 * TE_m)// for IR meco


static void ir_meco_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	ir_meco_init(data, dst);

	complex float* tmp_exp2   = md_alloc_sameplace(data->N, data->out2_dims, CFL_SIZE, dst); // for IR
	complex float* tmp_exp2_2 = md_alloc_sameplace(data->N, data->out2_dims, CFL_SIZE, dst); // for IR
	complex float* tmp_exp3   = md_alloc_sameplace(data->N, data->out3_dims, CFL_SIZE, dst); // for meco
	complex float* tmp_exp3_2 = md_alloc_sameplace(data->N, data->out3_dims, CFL_SIZE, dst); // for meco
	complex float* tmp_dst2   = md_alloc_sameplace(data->N, data->out2_dims, CFL_SIZE, dst); // for IR
	complex float* tmp_dst2_2 = md_alloc_sameplace(data->N, data->out2_dims, CFL_SIZE, dst); // for IR


	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// IR Look-Locker for Water
	// Ms_w
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->Ms_w, data->in_dims, src, CFL_SIZE);

	// M0_w
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->M0_w, data->in_dims, src, CFL_SIZE);

	// R1s_w
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->R1s_w, data->in_dims, src, CFL_SIZE);

	// exp(-t.*scaling_R1s_w*R1s_w):
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->R1s_w, -1.0*data->scaling_R1s_w);
	md_zmul2(data->N, data->out2_dims, data->out2_strs, tmp_exp2, data->map_strs, data->tmp_map, data->TI_strs, multiplace_read(data->TI, dst));
	md_zexp(data->N, data->out2_dims, tmp_exp2, tmp_exp2);

	// move R2s and fB0 to the front for the derivative calculation
	// R2s
	pos[COEFF_DIM] = 6;
	md_copy_block(data->N, pos, data->map_dims, data->R2s, data->in_dims, src, CFL_SIZE);

	// fB0
	pos[COEFF_DIM] = 7;
	md_copy_block(data->N, pos, data->map_dims, data->fB0, data->in_dims, src, CFL_SIZE);
	ir_meco_forw_fB0(data->linop_fB0, data->fB0, data->fB0); // Forward: convert from k-space to image-space

	// exp(1i*2*pi * LfB0 * fB0 .* TE)
	md_zmul2(data->N, data->out3_dims, data->out3_strs, tmp_exp3, data->map_strs, data->fB0, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out3_dims, tmp_exp3, tmp_exp3, 2.i * M_PI * data->scaling_fB0);
	md_zexp(data->N, data->out3_dims, tmp_exp3, tmp_exp3);

	// exp(-LR2s. * R2s.* TE)
	md_zmul2(data->N, data->out3_dims, data->out3_strs, tmp_exp3_2, data->map_strs, data->R2s, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out3_dims, tmp_exp3_2, tmp_exp3_2, -1.0 * data->scaling_R2s);
	md_zexp(data->N, data->out3_dims, tmp_exp3_2, tmp_exp3_2);

	// exp(-LR2s. * R2s.* TE) .* exp(1i*2*pi * LfB0 * fB0 .* TE)
	md_zmul(data->N, data->out3_dims, tmp_exp3_2, tmp_exp3_2, tmp_exp3);

	// dM0_w = exp(-TI.*scaling_R1s_w*R1s_w) x exp(1i*2*pi * LfB0 * fB0 .* TE) x exp(-LR2s. * R2s.* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dM0_w, data->out2_strs, tmp_exp2, data->out3_strs, tmp_exp3_2);

	// dMs_w: exp(1i*2*pi * LfB0 * fB0 .* TE) x exp(-LR2s. * R2s.* TE) - exp(-TI.*scaling_R1s_w*R1s_w) x exp(1i*2*pi * LfB0 * fB0 .* TE) x exp(-LR2s. * R2s.* TE)
	md_zsub2(data->N, data->out_dims, data->out_strs, data->dMs_w, data->out3_strs, tmp_exp3_2, data->out_strs, data->dM0_w);

	// dM0_w: -Lw0 .* exp(-TI.*scaling_R1s_w*R1s_w) x exp(1i*2*pi * LfB0 * fB0 .* TE) x exp(-LR2s. * R2s.* TE)
	md_zsmul(data->N, data->out_dims, data->dM0_w, data->dM0_w, -1.0*data->scaling_M0_w);

	// scaling_M0_w.*M0_w
	md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->M0_w, data->scaling_M0_w);

	// Ms_w + scaling_M0_w*M0_w
	md_zadd(data->N, data->map_dims, data->tmp_map, data->Ms_w, data->tmp_map);

	// (Ms_w + scaling_M0_w*M0_w).*exp(-TI.*scaling_R1s_w*R1s_w)
	md_zmul2(data->N, data->out2_dims, data->out2_strs, tmp_dst2, data->map_strs, data->tmp_map, data->out2_strs, tmp_exp2);

	// dR1s_w: LR1s_w .* TI x (Ms_w + scaling_M0_w*M0_w).*exp(-TI.*scaling_R1s_w*R1s_w) x exp(1i*2*pi * LfB0 * fB0 .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1s_w, data->out2_strs, tmp_dst2, data->out3_strs, tmp_exp3_2);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1s_w, data->out_strs, data->dR1s_w, data->TI_strs, multiplace_read(data->TI, dst));
	md_zsmul(data->N, data->out_dims, data->dR1s_w, data->dR1s_w, data->scaling_R1s_w);

	// Ms_w -(Ms_w + scaling_M0_w*M0_w).*exp(-t.*scaling_R1s_w*R1s_w)
	md_zsub2(data->N, data->out2_dims, data->out2_strs, tmp_dst2, data->map_strs, data->Ms_w, data->out2_strs, tmp_dst2);

	// IR Look-Locker for Fat

	// Ms_f
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->map_dims, data->Ms_f, data->in_dims, src, CFL_SIZE);

	// M0_f
	pos[COEFF_DIM] = 4;
	md_copy_block(data->N, pos, data->map_dims, data->M0_f, data->in_dims, src, CFL_SIZE);

	// R1s_f
	pos[COEFF_DIM] = 5;
	md_copy_block(data->N, pos, data->map_dims, data->R1s_f, data->in_dims, src, CFL_SIZE);

	// exp(-t.*scaling_R1s_f*R1s_f):
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->R1s_f, -1.0*data->scaling_R1s_f);
	md_zmul2(data->N, data->out2_dims, data->out2_strs, tmp_exp2, data->map_strs, data->tmp_map, data->TI_strs, multiplace_read(data->TI, dst));
	md_zexp(data->N, data->out2_dims, tmp_exp2, tmp_exp2);

	// exp(-TI.*scaling_R1s_f*R1s_f) x exp(1i*2*pi * LfB0 * fB0 .* TE) x exp(-LR2s. * R2s.* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dM0_f, data->out2_strs, tmp_exp2, data->out3_strs, tmp_exp3_2);

	// dMs_f: exp(1i*2*pi * LfB0 * fB0 .* TE) - exp(-TI.*scaling_R1s_f*R1s_f) x exp(1i*2*pi * LfB0 * fB0 .* TE)
	md_zsub2(data->N, data->out_dims, data->out_strs, data->dMs_f, data->out3_strs, tmp_exp3_2, data->out_strs, data->dM0_f);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dMs_f, data->out_strs, data->dMs_f, data->TE_strs, multiplace_read(data->cshift, dst));
	md_zsmul(data->N, data->out_dims, data->dMs_f, data->dMs_f, data->scaling_F);

	//dM0_f: -LF.*Lf0 .* exp(-TI.*scaling_R1s_f*R1s_f) x exp(1i*2*pi * LfB0 * fB0 .* TE). * zm
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dM0_f, data->out_strs, data->dM0_f, data->TE_strs, multiplace_read(data->cshift, dst));
	md_zsmul(data->N, data->out_dims, data->dM0_f, data->dM0_f, -1.0*data->scaling_F * data->scaling_M0_f);

	// scaling_M0_f.*M0_f
	md_zsmul2(data->N, data->map_dims, data->map_strs, data->tmp_map, data->map_strs, data->M0_f, data->scaling_M0_f);

	// Ms_f + scaling_M0_f*M0_f
	md_zadd(data->N, data->map_dims, data->tmp_map, data->Ms_f, data->tmp_map);

	// (Ms_f + scaling_M0_f*M0_f).*exp(-t.*scaling_R1s_f*R1s_f)
	md_zmul2(data->N, data->out2_dims, data->out2_strs, tmp_dst2_2, data->map_strs, data->tmp_map, data->out2_strs, tmp_exp2);

	// dR1s_f: L_F .* LR1s_f .* TI x zm x (Ms_f + scaling_M0_f*M0_f).*exp(-TI.*scaling_R1s_f*R1s_f) x exp(1i*2*pi * LfB0 * fB0 .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1s_f, data->out2_strs, tmp_dst2_2, data->TE_strs, multiplace_read(data->cshift, dst));
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1s_f, data->out_strs, data->dR1s_f, data->TI_strs, multiplace_read(data->TI, dst));
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1s_f, data->out_strs, data->dR1s_f, data->out3_strs, tmp_exp3_2);
	md_zsmul(data->N, data->out_dims, data->dR1s_f, data->dR1s_f, data->scaling_R1s_f * data->scaling_F);

	// Ms_f -(Ms_f + scaling_M0_f*M0_f).*exp(-t.*scaling_R1s_f*R1s_f)
	md_zsub2(data->N, data->out2_dims, data->out2_strs, tmp_dst2_2, data->map_strs, data->Ms_f, data->out2_strs, tmp_dst2_2);

	// dst = F .* cshift
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out2_strs, tmp_dst2_2, data->TE_strs, multiplace_read(data->cshift, dst));
	md_zsmul(data->N, data->out_dims, dst, dst, data->scaling_F);

	// Signal Model: dst = (W + F .* cshift) x exp(1i*2*pi * LfB0 * fB0 .* TE) x exp(-LR2s. * R2s.* TE)
	md_zadd2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, data->out2_strs, tmp_dst2);
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out_strs, dst, data->out3_strs, tmp_exp3_2);

	// dR2s:
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR2s, data->out_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dR2s, data->dR2s, -data->scaling_R2s);

	// dfB_0:
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dfB0, data->out_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dfB0, data->dfB0, I * 2.* M_PI * data->scaling_fB0);

	md_free(tmp_dst2_2);
	md_free(tmp_dst2);
	md_free(tmp_exp3_2);
	md_free(tmp_exp3);
	md_free(tmp_exp2_2);
	md_free(tmp_exp2);
}

static void ir_meco_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	//dMs_w
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dMs_w);

	//dM0_w
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dM0_w);

	//dR1s_w
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR1s_w);

	//dMs_f
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dMs_f);

	//dM0_f
	pos[COEFF_DIM] = 4;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dM0_f);

	//dR1s_f
	pos[COEFF_DIM] = 5;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR1s_f);

        //dR2s
	pos[COEFF_DIM] = 6;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR2s);

	//dfB0
	pos[COEFF_DIM] = 7;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	ir_meco_forw_fB0(data->linop_fB0, data->tmp_map, data->tmp_map); // Forward: convert from k-space to image-space
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dfB0);
}

static void ir_meco_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// dst[0] = sum(sum (conj(Ms_w') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dMs_w);
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[1] = sum(sum (conj(M0_w') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dM0_w);
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[2] = sum(sum (conj(R1s_w') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR1s_w);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);


	// dst[3] = sum(sum (conj(Ms_f') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dMs_f);
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[4] = sum(sum (conj(M0_f') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dM0_f);
	pos[COEFF_DIM] = 4;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[5] = sum(sum (conj(R1s_f') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR1s_f);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	pos[COEFF_DIM] = 5;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[6] = sum(sum (conj(R2s') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR2s);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	pos[COEFF_DIM] = 6;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[7] = sum(sum (conj(fB0) * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dfB0);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	ir_meco_back_fB0(data->linop_fB0, data->tmp_map, data->tmp_map); // Backward: convert from image-space to k-space
	pos[COEFF_DIM] = 7;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}

// Calculate water-only Model:

// T1 and T2*: [Ms - (Ms + L_M0.*M0).* exp(-t * L_R1s.* R1s)] * exp(1i*2*pi * L_fB0 * fB0 .* TE) * exp(-L_R2s.* R2s .* TE)

static void ir_meco_w_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	ir_meco_init(data, dst);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	complex float* tmp_exp2   = md_alloc_sameplace(data->N, data->out2_dims, CFL_SIZE, dst); // for IR
	complex float* tmp_exp2_2 = md_alloc_sameplace(data->N, data->out2_dims, CFL_SIZE, dst); // for IR
	complex float* tmp_exp3   = md_alloc_sameplace(data->N, data->out3_dims, CFL_SIZE, dst); // for meco
	complex float* tmp_exp3_2 = md_alloc_sameplace(data->N, data->out3_dims, CFL_SIZE, dst); // for meco

	// Ms
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->Ms_w, data->in_dims, src, CFL_SIZE);

	// M0
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->M0_w, data->in_dims, src, CFL_SIZE);

	// R1*
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->R1s_w, data->in_dims, src, CFL_SIZE);

	// R2*
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->map_dims, data->R2s, data->in_dims, src, CFL_SIZE);

	// fB0
	pos[COEFF_DIM] = 4;
	md_copy_block(data->N, pos, data->map_dims, data->fB0, data->in_dims, src, CFL_SIZE);
	ir_meco_forw_fB0(data->linop_fB0, data->fB0, data->fB0); // Forward: convert from k-space to image-space

	// tmp_exp2 = exp(-t.*scaling_R1s*R1s):
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->R1s_w, -1.0*data->scaling_R1s_w);
	md_zmul2(data->N, data->out2_dims, data->out2_strs, tmp_exp2, data->map_strs, data->tmp_map, data->TI_strs, multiplace_read(data->TI, dst));
	md_zexp(data->N, data->out2_dims, tmp_exp2, tmp_exp2);

	// exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zmul2(data->N, data->out3_dims, data->out3_strs, tmp_exp3, data->map_strs, data->fB0, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out3_dims, tmp_exp3, tmp_exp3, 2.i * M_PI * data->scaling_fB0);
	md_zexp(data->N, data->out3_dims, tmp_exp3, tmp_exp3);

	// exp(-L_R2s.* R2s .* TE)
	md_zmul2(data->N, data->out3_dims, data->out3_strs, tmp_exp3_2, data->map_strs, data->R2s, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out3_dims, tmp_exp3_2, tmp_exp3_2, -1.0*data->scaling_R2s);
	md_zexp(data->N, data->out3_dims, tmp_exp3_2, tmp_exp3_2);

	// tmp_exp3_2 = exp(-L_R2s.* R2s .* TE).* exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zmul(data->N, data->out3_dims, tmp_exp3_2, tmp_exp3, tmp_exp3_2);

	// dM0 = exp(-L_R2s.* R2s .* TE) .* exp(1i*2*pi * L_fB0 * fB0 .* TE) .* exp(-t.*scaling_R1s*R1s)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dM0_w, data->out2_strs, tmp_exp2, data->out3_strs, tmp_exp3_2);

	// Ms + L_M0 .* M0
	md_zsmul(data->N, data->map_dims, data->tmp_map, data->M0_w, data->scaling_M0_w);
	md_zadd(data->N, data->map_dims, data->tmp_map, data->Ms_w, data->tmp_map);

	// dR1s: L_R1s.* t.* (Ms + L_M0.*M0).* exp(-t * L_R1s.* R1s) * exp(1i*2*pi * L_fB0 * fB0 .* TE) * exp(-L_R2s.* R2s .* TE)
	// tmp_exp2_2 = (Ms + L_M0.*M0).* exp(-t * L_R1s.* R1s)
	md_zmul2(data->N, data->out2_dims, data->out2_strs, tmp_exp2_2, data->out2_strs, tmp_exp2, data->map_strs, data->tmp_map);

	// tmp_exp2 = TI. * (Ms + L_M0.*M0).* exp(-t * L_R1s.* R1s)
	md_zmul2(data->N, data->out2_dims, data->out2_strs, tmp_exp2, data->out2_strs, tmp_exp2_2, data->TI_strs, multiplace_read(data->TI, dst));

	// data->dR1s = TI. * (Ms + L_M0.*M0).* exp(-t * L_R1s.* R1s) .* exp(-L_R2s.* R2s .* TE).* exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR1s_w, data->out2_strs, tmp_exp2, data->out3_strs, tmp_exp3_2);

	md_zsmul(data->N, data->out_dims, data->dR1s_w, data->dR1s_w, data->scaling_R1s_w);

	// tmp_exp2_2 = Ms - (Ms + L_M0 .* M0) .* exp(-t.*scaling_R1s*R1s)
	md_zsub2(data->N, data->out2_dims, data->out2_strs, tmp_exp2_2, data->map_strs, data->Ms_w, data->out2_strs, tmp_exp2_2);

	// Model: dst = (Ms - (Ms + L_M0 .* M0) .* exp(-t.*scaling_R1s*R1s)).* exp(i * 2pi * L_fB0. *fB0 * TE).* exp(-L_R2s.* R2s .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->out2_strs, tmp_exp2_2, data->out3_strs, tmp_exp3_2);

	// dMs: (1 - exp(-t * L_R1s.* R1s)) * exp(1i*2*pi * L_fB0 * fB0 .* TE) * exp(-L_R2s.* R2s .* TE)
	md_zsub2(data->N, data->out_dims, data->out_strs, data->dMs_w, data->out3_strs, tmp_exp3_2, data->out_strs, data->dM0_w);

	// dM0: - L_M0 .* exp(-t.*scaling_R1s*R1s)).* exp(i * 2pi * L_fB0. *fB0 * TE).* exp(-L_R2s.* R2s .* TE)
	md_zsmul(data->N, data->out_dims, data->dM0_w, data->dM0_w, -data->scaling_M0_w);

	// dR2s: -L_R2s.* TE .*(Ms - (Ms + L_M0 .* M0) .* exp(-t.*scaling_R1s*R1s)).* exp(i * 2pi * L_fB0. *fB0 * TE).* exp(-L_R2s.* R2s .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR2s, data->out_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dR2s, data->dR2s, -data->scaling_R2s);


	// dfB0:  i * 2pi * L_fB0. TE. *(Ms - (Ms + L_M0 .* M0) .* exp(-t.*scaling_R1s*R1s)).* exp(i * 2pi * L_fB0. *fB0 * TE).* exp(-L_R2s.* R2s .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dfB0, data->out_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dfB0, data->dfB0, data->scaling_fB0 * 2.i * M_PI);

	md_free(tmp_exp3_2);
	md_free(tmp_exp3);
	md_free(tmp_exp2_2);
	md_free(tmp_exp2);
}


static void ir_meco_w_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	//dMs
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dMs_w);

	//dM0
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dM0_w);

	//dR1s
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR1s_w);

	//dR2s
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR2s);

	//dfB0
	pos[COEFF_DIM] = 4;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	ir_meco_forw_fB0(data->linop_fB0, data->tmp_map, data->tmp_map); // Forward: convert from k-space to image-space
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dfB0);
}

static void ir_meco_w_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// dst[0] = sum(sum (conj(Ms_w') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dMs_w);
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[1] = sum(sum (conj(M0_w') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dM0_w);
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[2] = sum(sum (conj(R1s_w') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR1s_w);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[3] = sum(sum (conj(R2s') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR2s);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[4] = sum(sum (conj(fB0') * src, TE), TI)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dfB0);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	ir_meco_back_fB0(data->linop_fB0, data->tmp_map, data->tmp_map); // Backward: convert from image-space to k-space
	pos[COEFF_DIM] = 4;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}




static void meco_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	ir_meco_init(data, dst);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// Water
	// W
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->Ms_w, data->in_dims, src, CFL_SIZE);

	// Fat
	// F
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->Ms_f, data->in_dims, src, CFL_SIZE);

	// R2*
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->R2s, data->in_dims, src, CFL_SIZE);

	// fB0
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->map_dims, data->fB0, data->in_dims, src, CFL_SIZE);

	ir_meco_forw_fB0(data->linop_fB0, data->fB0, data->fB0);

	// exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dMs_w, data->map_strs, data->fB0, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dMs_w, data->dMs_w, 2.i * M_PI * data->scaling_fB0);

	// dW: exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zexp(data->N, data->out_dims, data->dMs_w, data->dMs_w);

	// exp(-L_R2s.* R2s .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dMs_f, data->map_strs, data->R2s, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dMs_f, data->dMs_f, -data->scaling_R2s);

	// dF: exp(-L_R2s.* R2s .* TE)
	md_zexp(data->N, data->out_dims, data->dMs_f, data->dMs_f);

	// dW: exp(-L_R2s.* R2s .* TE) .* exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zmul(data->N, data->out_dims, data->dMs_w, data->dMs_w, data->dMs_f);

	// L_F .* F .* zm
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Ms_f, data->TE_strs, multiplace_read(data->cshift, dst));
	md_zsmul(data->N, data->out_dims, dst, dst, data->scaling_F);

	// Model: (W + L_F .*F .* zm).* exp(i * 2pi * L_fB0. *fB0 * TE).* exp(-L_R2s.* R2s .* TE)
	md_zadd2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Ms_w, data->out_strs, dst);

	md_zmul(data->N, data->out_dims, dst, data->dMs_w, dst);

	// dF = L_F .* zm. * exp(i * 2pi * L_fB0. *fB0 * TE).* exp(-L_R2s.* R2s .* TE)
	md_zsmul(data->N, data->out_dims, data->dMs_f, data->dMs_w, data->scaling_F);

	md_zmul2(data->N, data->out_dims, data->out_strs, data->dMs_f, data->out_strs, data->dMs_f, data->TE_strs, multiplace_read(data->cshift, dst));

	// dfB0:
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dR2s, data->out_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dfB0, data->dR2s, data->scaling_fB0 * 2.i * M_PI);

	// dR2s:
	md_zsmul(data->N, data->out_dims, data->dR2s, data->dR2s, -data->scaling_R2s);
}


static void meco_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	//dW
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dMs_w);

	//dF
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dMs_f);

	//dR2s
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dR2s);

	//dfB0
	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	ir_meco_forw_fB0(data->linop_fB0, data->tmp_map, data->tmp_map);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dfB0);
}

static void meco_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// dst[0] =sum (conj(Ms_w') * src, TE)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dMs_w);
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[1] = sum (conj(Ms_f') * src, TE)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dMs_f);
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[2] = sum (conj(R2s') * src, TE)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dR2s);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[3] = sum (conj(fB0') * src, TE)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dfB0);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

	ir_meco_back_fB0(data->linop_fB0, data->tmp_map, data->tmp_map);

	pos[COEFF_DIM] = 3;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}

static void meco_fun2(const nlop_data_t* _data, complex float* dst, const complex float* src) // triple echo
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	ir_meco_init(data, dst);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// Water
	// W
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->Ms_w, data->in_dims, src, CFL_SIZE);

	// Fat
	// F
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->Ms_f, data->in_dims, src, CFL_SIZE);

	// fB0
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->fB0, data->in_dims, src, CFL_SIZE);

	ir_meco_forw_fB0(data->linop_fB0, data->fB0, data->fB0); // from k-space to image space

	// exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dMs_w, data->map_strs, data->fB0, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dMs_w, data->dMs_w, 2.i * M_PI * data->scaling_fB0);

	// dW: exp(1i*2*pi * L_fB0 * fB0 .* TE)
	md_zexp(data->N, data->out_dims, data->dMs_w, data->dMs_w);

	// L_F .* F .* zm
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Ms_f, data->TE_strs, multiplace_read(data->cshift, dst));
	md_zsmul(data->N, data->out_dims, dst, dst, data->scaling_F);

	// Model: (W + L_F .*F .* zm).* exp(i * 2pi * L_fB0. *fB0 * TE)
	md_zadd2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Ms_w, data->out_strs, dst);
	md_zmul(data->N, data->out_dims, dst, data->dMs_w, dst);

	// dF = L_F .* zm. * exp(i * 2pi * L_fB0. *fB0 * TE)
	md_zsmul(data->N, data->out_dims, data->dMs_f, data->dMs_w, data->scaling_F);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dMs_f, data->out_strs, data->dMs_f, data->TE_strs, multiplace_read(data->cshift, dst));

	// dfB0:
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dfB0, data->out_strs, dst, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, data->dfB0, data->dfB0, data->scaling_fB0 * 2.i * M_PI);
}


static void meco_der2(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	//dW
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dMs_w);

	//dF
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dMs_f);

	//dfB0
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	ir_meco_forw_fB0(data->linop_fB0, data->tmp_map, data->tmp_map);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->dfB0);
}

static void meco_adj2(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// dst[0] =sum (conj(Ms_w') * src, TE)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dMs_w);
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[1] = sum (conj(Ms_f') * src, TE)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dMs_f);
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// dst[2] = sum (conj(fB0') * src, TE)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->dfB0);
	md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);
	ir_meco_back_fB0(data->linop_fB0, data->tmp_map, data->tmp_map);
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}


static void ir_meco_del(const nlop_data_t* _data)
{
	struct ir_meco_s* data = CAST_DOWN(ir_meco_s, _data);

	md_free(data->Ms_w);
	md_free(data->M0_w);
	md_free(data->R1s_w);

	md_free(data->R2s);
	md_free(data->fB0);

	md_free(data->tmp_map);

	md_free(data->dMs_w);
	md_free(data->dM0_w);
	md_free(data->dR1s_w);

	md_free(data->dR2s);

	md_free(data->dfB0);

	md_free(data->Ms_f);
	md_free(data->M0_f);
	md_free(data->R1s_f);
	md_free(data->dMs_f);
	md_free(data->dM0_f);
	md_free(data->dR1s_f);

	xfree(data->map_dims);
	xfree(data->TI_dims);
	xfree(data->TE_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);
	xfree(data->out2_dims);
	xfree(data->out3_dims);

	xfree(data->map_strs);
	xfree(data->TI_strs);
	xfree(data->TE_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);
	xfree(data->out2_strs);
	xfree(data->out3_strs);

	linop_free(data->linop_fB0);

	multiplace_free(data->TI);
	multiplace_free(data->TE);
	multiplace_free(data->cshift);

	xfree(data);
}

struct nlop_s* nlop_ir_meco_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N],
				const complex float* TI, const long TE_dims[N], const complex float* TE, const float* scale_fB0, const float* scale)
{

	PTR_ALLOC(struct ir_meco_s, data);
	SET_TYPEID(ir_meco_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	// IR + meco
	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(long[N], ntidims);
	md_copy_dims(N, *ntidims, TI_dims);
	data->TI_dims = *PTR_PASS(ntidims);

	PTR_ALLOC(long[N], ntedims);
	md_copy_dims(N, *ntedims, TE_dims);
	data->TE_dims = *PTR_PASS(ntedims);

	PTR_ALLOC(long[N], nmstr);
	md_calc_strides(N, *nmstr, map_dims, CFL_SIZE);
	data->map_strs = *PTR_PASS(nmstr);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, out_dims, CFL_SIZE);
	data->out_strs = *PTR_PASS(nostr);

	PTR_ALLOC(long[N], nistr);
	md_calc_strides(N, *nistr, in_dims, CFL_SIZE);
	data->in_strs = *PTR_PASS(nistr);

	PTR_ALLOC(long[N], ntistr);
	md_calc_strides(N, *ntistr, TI_dims, CFL_SIZE);
	data->TI_strs = *PTR_PASS(ntistr);

	PTR_ALLOC(long[N], ntestr);
	md_calc_strides(N, *ntestr, TE_dims, CFL_SIZE);
	data->TE_strs = *PTR_PASS(ntestr);


	// for IR
	long out2_dims[N];
	md_select_dims(N, ~CSHIFT_FLAG, out2_dims, out_dims);

	PTR_ALLOC(long[N], nodims2);
	md_copy_dims(N, *nodims2, out2_dims);
	data->out2_dims = *PTR_PASS(nodims2);

	PTR_ALLOC(long[N], nostr2);
	md_calc_strides(N, *nostr2, out2_dims, CFL_SIZE);
	data->out2_strs = *PTR_PASS(nostr2);

	// for meco
	long out3_dims[N];
	md_select_dims(N, ~TE_FLAG, out3_dims, out_dims);

	PTR_ALLOC(long[N], nodims3);
	md_copy_dims(N, *nodims3, out3_dims);
	data->out3_dims = *PTR_PASS(nodims3);

	PTR_ALLOC(long[N], nostr3);
	md_calc_strides(N, *nostr3, out3_dims, CFL_SIZE);
	data->out3_strs = *PTR_PASS(nostr3);


	data->N = N;
	data->Ms_w = NULL;
	data->M0_w = NULL;
	data->R1s_w = NULL;
	data->Ms_f = NULL;
	data->M0_f = NULL;
	data->R1s_f = NULL;
	data->fB0 = NULL;
	data->R2s = NULL;
	data->tmp_map = NULL;
	data->dMs_w = NULL;
	data->dM0_w = NULL;
	data->dR1s_w = NULL;
	data->dMs_f = NULL;
	data->dM0_f = NULL;
	data->dR1s_f = NULL;
	data->dR2s = NULL;
	data->dfB0 = NULL;

#if 1
	// weight on alpha
	long w_dims[N];
	md_select_dims(N, FFT_FLAGS, w_dims, map_dims);
	double a = scale_fB0[0];
	double b = scale_fB0[1];

	complex float* weights = md_alloc(N, w_dims, CFL_SIZE);
	noir_calc_weights(a, b, w_dims, weights);

	const struct linop_s* linop_wghts = linop_cdiag_create(N, map_dims, FFT_FLAGS, weights);
	const struct linop_s* linop_ifftc = linop_ifftc_create(N, map_dims, FFT_FLAGS);

	md_free(weights);

	data->linop_fB0 = linop_chain(linop_wghts, linop_ifftc); // IFFT(W.* \hat{x_{k}})

	linop_free(linop_wghts);
	linop_free(linop_ifftc);
#endif

	data->TI = multiplace_move(N, TI_dims, CFL_SIZE, TI);
	data->TE = multiplace_move(N, TE_dims, CFL_SIZE, TE);

	// Calculate cshift

	complex float* cshift = md_alloc(N, TE_dims, CFL_SIZE);

	enum fat_spec fat_spec = FAT_SPEC_1;
	ir_meco_calc_fat_modu(N, TE_dims, TE, cshift, fat_spec);
	data->cshift = multiplace_move(N, TE_dims, CFL_SIZE, cshift);

	md_free(cshift);

	data->scaling_M0_w   = 1.;
	data->scaling_R1s_w  = 1.;
	data->scaling_F      = 1.;
	data->scaling_M0_f   = 1.;
	data->scaling_R1s_f  = 1.;
	data->scaling_R2s    = 1.;
	data->scaling_fB0    = 1.;

	if (3 == in_dims[COEFF_DIM]) { // W, F, fB0

		debug_printf(DP_DEBUG1, "MODEL: W, F, fB0\n");

		data->scaling_F      = scale[1];
		data->scaling_R2s    = 1.;	// Not used
		data->scaling_fB0    = scale[2];

		return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), meco_fun2, meco_der2, meco_adj2, NULL, NULL, ir_meco_del);

	} else if (4 == in_dims[COEFF_DIM]) { // W, F, R2*, fB0

		debug_printf(DP_DEBUG1, "MODEL: W, F, R2*, fB0\n");

		data->scaling_F      = scale[1];
		data->scaling_R2s    = scale[2];
		data->scaling_fB0    = scale[3];

		return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), meco_fun, meco_der, meco_adj, NULL, NULL, ir_meco_del);

	} else if (5 == in_dims[COEFF_DIM]) { // Ms, M0, R1*, R2*, fB0

		debug_printf(DP_DEBUG1, "MODEL: Ms, M0, R1*, R2*, fB0\n");

		data->scaling_F      = 1.;	// Not used
		data->scaling_R2s    = scale[3];
		data->scaling_fB0    = scale[4];

		return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), ir_meco_w_fun, ir_meco_w_der, ir_meco_w_adj, NULL, NULL, ir_meco_del);

	} else { // Ms_w, M0_w, R1*_w, Ms_f, M0_f, R1*_f, R2*, fB0

		debug_printf(DP_DEBUG1, "MODEL: Ms_w, M0_w, R1*_w, Ms_f, M0_f, R1*_f, R2*, fB0\n");

		// FIXME: Remove hard-coded value of 0.5 here! Pass as additional option to moba?
		data->scaling_F      = 0.5;
		data->scaling_R2s    = scale[6];
		data->scaling_fB0    = scale[7];

		return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), ir_meco_fun, ir_meco_der, ir_meco_adj, NULL, NULL, ir_meco_del);
	}
}


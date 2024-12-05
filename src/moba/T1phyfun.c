/* Copyright 2022. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Xiaoqing Wang, Nick Scholand, Martin Uecker
 */

#include <complex.h>
#include <math.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/multiplace.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "nlops/nlop.h"

#include "moba/moba.h"

#include "noir/utils.h"

#include "T1phyfun.h"

struct T1_phy_s {

	INTERFACE(nlop_data_t);

	int N;

	const long* map_dims;
	const long* TI_dims;
	const long* in_dims;
	const long* out_dims;

	const long* map_strs;
	const long* TI_strs;
	const long* in_strs;
	const long* out_strs;

	// Parameter maps
	complex float* M0;
	complex float* R1;
	complex float* alpha;

	// FIXME: temporary storage should be allocated when used
	complex float* tmp_map;
	complex float* tmp_R1s;
	complex float* tmp_map1;
	complex float* tmp_ones;
	complex float* tmp_exp;

	complex float* tmp_dM0;
	complex float* tmp_dR1;
	complex float* tmp_dalpha;

	struct multiplace_array_s* TI;

	const struct linop_s* linop_alpha;

	float scaling_alpha;
	float r1p_nom;
};

DEF_TYPEID(T1_phy_s);


const struct linop_s* T1_get_alpha_trafo(struct nlop_s* op)
{
	struct T1_phy_s* data = CAST_DOWN(T1_phy_s, nlop_get_data(op));

	return data->linop_alpha;
}

void T1_forw_alpha(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_forward_unchecked(op, dst, src);
}

void T1_back_alpha(const struct linop_s* op, complex float* dst, const complex float* src)
{
	linop_adjoint_unchecked(op, dst, src);
}

/**
 * Readout relaxation rate for reparameterized Look-Locker model
 * Roeloffs, V., Wang, X., Sumpf, T.J., Untenberger, M., Voit, D. and Frahm, J. (2016),
 * Model-based reconstruction for T1 mapping using single-shot inversion-recovery radial FLASH.
 * Int. J. Imaging Syst. Technol., 26: 254-263. https://doi.org/10.1002/ima.22196z
 * @param tr 	repetition time [s]
 * @param angle flip angle [rad]
 */
float read_relax(float tr, float angle)
{
	return -1. / tr * logf(cosf(angle));
}

// Calculate Model: M0 * (R1/(R1 + alpha) - (1 + R1/(R1 + alpha)) * exp(-t.*(R1 + alpha)))
static void T1_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_phy_s* data = CAST_DOWN(T1_phy_s, _data);

	if (NULL == data->R1) {

		data->R1 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->M0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->alpha = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
        	data->tmp_map1 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->tmp_ones = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->tmp_R1s = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->tmp_exp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
		data->tmp_dM0 = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
		data->tmp_dR1 = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
		data->tmp_dalpha = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
	}

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

        float reg_parameter = 1e-8;

	// M0
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->M0, data->in_dims, src, CFL_SIZE);

	// R1
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->R1, data->in_dims, src, CFL_SIZE);

	// alpha
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->alpha, data->in_dims, src, CFL_SIZE);

	T1_forw_alpha(data->linop_alpha, data->tmp_map, data->alpha);

	// R1p_nom = -ln(cos(fa_nom))/tr = alpha_nom -> Sobolev on r1p around 0 instead of 1
	md_zfill(data->N, data->map_dims, data->tmp_ones, data->r1p_nom);

	// R1s = R1 + (1 + alpha * scaling_alpha)
	md_zsmul(data->N, data->map_dims, data->tmp_R1s, data->tmp_map, data->scaling_alpha);
	md_zadd(data->N, data->map_dims, data->tmp_R1s, data->tmp_ones, data->tmp_R1s);
	md_zadd(data->N, data->map_dims, data->tmp_R1s, data->R1, data->tmp_R1s);

	// exp(-t.* (R1 + alpha * scaling_alpha)):
        md_zsmul(data->N, data->map_dims, data->tmp_map, data->tmp_R1s, -1.0);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->map_strs, data->tmp_map, data->TI_strs, multiplace_read(data->TI, dst));

	md_zexp(data->N, data->out_dims, data->tmp_exp, data->tmp_exp);

	//1 + R1/R1s
	md_zdiv_reg(data->N, data->map_dims, data->tmp_map, data->R1, data->tmp_R1s, reg_parameter); // tmp_map = R1/R1s
	md_zfill(data->N, data->map_dims, data->tmp_ones, 1.0);
        md_zadd(data->N, data->map_dims, data->tmp_map1, data->tmp_ones, data->tmp_map); // tmp_map1 = 1 + R1/R1s

	// (1 + R1/R1s).*exp(-t.* R1s)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dalpha, data->map_strs, data->tmp_map1, data->out_strs, data->tmp_exp);

	//Model: M0*( R1/R1s -(1 + R1/R1s).*exp(-t.* R1s))
	md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_dM0, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dalpha);
        md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->M0, data->out_strs, data->tmp_dM0);

	// Calculating derivatives
	// t * exp(-t*R1s) * (1 + R1/R1s)
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dalpha, data->out_strs, data->tmp_dalpha, data->TI_strs, multiplace_read(data->TI, dst));

	// R1 / R1s.^2
        md_zmul(data->N, data->map_dims, data->tmp_map, data->tmp_R1s, data->tmp_R1s);
	md_zdiv_reg(data->N, data->map_dims, data->tmp_map, data->R1, data->tmp_map, reg_parameter);

	// R1 / R1s.^2 .* (exp(-t * R1s) - 1)
	md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_exp, data->out_strs, data->tmp_exp, data->map_strs, data->tmp_ones);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1, data->map_strs, data->tmp_map, data->out_strs, data->tmp_exp);

	// alpha'
	md_zadd(data->N, data->out_dims, data->tmp_dalpha, data->tmp_dalpha, data->tmp_dR1);
        md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dalpha, data->map_strs, data->M0, data->out_strs, data->tmp_dalpha);

	// R1'
	md_zdiv_reg(data->N, data->map_dims, data->tmp_map, data->tmp_ones, data->tmp_R1s, reg_parameter);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1, data->map_strs, data->tmp_map, data->out_strs, data->tmp_exp);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1, data->map_strs, data->M0, data->out_strs, data->tmp_dR1);
	md_zsub(data->N, data->out_dims, data->tmp_dR1, data->tmp_dalpha, data->tmp_dR1);

	// alpha' * scaling_alpha
	md_zsmul(data->N, data->out_dims, data->tmp_dalpha, data->tmp_dalpha, data->scaling_alpha);

        // FIXME: Precalculate derivatives here and perform md_ztenmul only in operators below -> potential speed up
}

static void T1_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct T1_phy_s* data = CAST_DOWN(T1_phy_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// tmp = dR1
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);

	//const complex float* tmp_M0 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = R1' * dR1
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dR1);

	// tmp = dM0
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	//const complex float* tmp_Mss = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	// dst = dst + dMss * Mss'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dM0);

	// tmp =  dalpha
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->tmp_map, data->in_dims, src, CFL_SIZE);
	//const complex float* tmp_alpha = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	T1_forw_alpha(data->linop_alpha, data->tmp_map, data->tmp_map);

	// dst = dst + dalpha * alpha'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->tmp_map, data->out_strs, data->tmp_dalpha);
}

static void T1_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct T1_phy_s* data = CAST_DOWN(T1_phy_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// sum (conj(M0') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dR1);
	//md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

	// dst[1] = sum (conj(M0') * src, t)
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// sum (conj(Mss') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dM0);

	// dst[0] = sum (conj(Mss') * src, t)
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);

	// sum (conj(alpha') * src, t)
	md_clear(data->N, data->map_dims, data->tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, data->tmp_map, data->out_strs, src, data->out_strs, data->tmp_dalpha);

        // Real constraint through adjoint derivative operator? -> breaks scalar product test!
        // md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

        T1_back_alpha(data->linop_alpha, data->tmp_map, data->tmp_map);

	// dst[2] = sum (conj(alpha') * src, t)
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, data->tmp_map, CFL_SIZE);
}

static void T1_del(const nlop_data_t* _data)
{
	struct T1_phy_s* data = CAST_DOWN(T1_phy_s, _data);

	md_free(data->R1);
	md_free(data->M0);
	md_free(data->alpha);

	multiplace_free(data->TI);

	md_free(data->tmp_map);
        md_free(data->tmp_R1s);
        md_free(data->tmp_map1);
	md_free(data->tmp_ones);
	md_free(data->tmp_exp);

	md_free(data->tmp_dM0);
	md_free(data->tmp_dR1);
	md_free(data->tmp_dalpha);

	xfree(data->map_dims);
	xfree(data->TI_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->map_strs);
	xfree(data->TI_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	linop_free(data->linop_alpha);

	xfree(data);
}


struct nlop_s* nlop_T1_phy_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N], const complex float* TI,  const struct moba_conf_s* config)
{
	PTR_ALLOC(struct T1_phy_s, data);
	SET_TYPEID(T1_phy_s, data);


	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

	PTR_ALLOC(long[N], ntidims);
	md_copy_dims(N, *ntidims, TI_dims);
	data->TI_dims = *PTR_PASS(ntidims);

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

	data->N = N;
	data->R1 = NULL;
	data->M0 = NULL;
	data->alpha = NULL;
	data->tmp_map = NULL;
        data->tmp_map1 = NULL;
	data->tmp_ones = NULL;
	data->tmp_R1s = NULL;
	data->tmp_exp = NULL;
	data->tmp_dM0 = NULL;
	data->tmp_dR1 = NULL;
	data->tmp_dalpha = NULL;
	data->TI = multiplace_move(N, TI_dims, CFL_SIZE, TI);


	// weight on alpha
	long w_dims[N];
	md_select_dims(N, FFT_FLAGS, w_dims, map_dims);

	complex float* weights = md_alloc(N, w_dims, CFL_SIZE);

	noir_calc_weights(config->other.b1_sobolev_a, config->other.b1_sobolev_b, w_dims, weights);

	const struct linop_s* linop_wghts = linop_cdiag_create(N, map_dims, FFT_FLAGS, weights);
	const struct linop_s* linop_ifftc = linop_ifftc_create(N, map_dims, FFT_FLAGS);

	data->linop_alpha = linop_chain_FF(linop_wghts, linop_ifftc);

	md_free(weights);

	data->scaling_alpha = config->other.scale[2];
	data->r1p_nom = read_relax(config->sim.seq.tr, DEG2RAD(CAST_UP(&config->sim.pulse.sinc)->flipangle));

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T1_fun, T1_der, T1_adj, NULL, NULL, T1_del);
}

/* Copyright 2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Xiaoqing Wang, Martin Uecker
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/version.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"

#include "nlops/nlop.h"

#include "T1fun.h"

struct T1_s {

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
	complex float* Mss;
	complex float* M0;
	complex float* R1s;

	complex float* tmp_dMss;
	complex float* tmp_dM0;
	complex float* tmp_dR1s;

	struct multiplace_array_s* TI;

	float scaling_M0;
	float scaling_R1s;
};

DEF_TYPEID(T1_s);

static void T1_init(struct T1_s* data, const void* arg)
{
	if (NULL != data->Mss)
		return;

	data->Mss = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
	data->M0 = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
	data->R1s = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, arg);
	data->tmp_dM0 = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
	data->tmp_dMss = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
	data->tmp_dR1s = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, arg);
}

// Calculate Model: Mss - (Mss + M0) * exp(-t.*R1s)
static void T1_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

	T1_init(data, dst);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_ones = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
	complex float* tmp_exp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);

	// Mss
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->Mss, data->in_dims, src, CFL_SIZE);

	// M0
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->M0, data->in_dims, src, CFL_SIZE);

	// R1s
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, data->R1s, data->in_dims, src, CFL_SIZE);
	if (!use_compat_to_version("v0.7.00"))
		md_zreal(data->N, data->map_dims, data->R1s, data->R1s);


	// -1*scaling_R1s.*R1s
	md_zsmul2(data->N, data->map_dims, data->map_strs, tmp_map, data->map_strs, data->R1s, -1.0*data->scaling_R1s);

	// exp(-t.*scaling_R1s*R1s):

	md_zmul2(data->N, data->out_dims, data->out_strs, tmp_exp, data->map_strs, tmp_map, data->TI_strs, multiplace_read(data->TI, dst));
	md_zexp(data->N, data->out_dims, tmp_exp, tmp_exp);

	// scaling_M0.*M0
	md_zsmul2(data->N, data->map_dims, data->map_strs, tmp_map, data->map_strs, data->M0, data->scaling_M0);

	// Mss + scaling_M0*M0
	md_zadd(data->N, data->map_dims, tmp_map, data->Mss, tmp_map);

	// (Mss + scaling_M0*M0).*exp(-t.*scaling_R1s*R1s)
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, tmp_map, data->out_strs, tmp_exp);

	// Mss -(Mss + scaling_M0*M0).*exp(-t.*scaling_R1s*R1s)
	md_zsub2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->Mss, data->out_strs, dst);

	// Calculating derivatives

	// M0' = -scaling_M0.*exp(-t.*scaling_R1s.*R1s)
	md_zsmul(data->N, data->out_dims, data->tmp_dM0, tmp_exp, -data->scaling_M0);

	// Mss' = 1 - exp(-t.*scaling_R1s.*R1s)
	md_zfill(data->N, data->map_dims, tmp_ones, 1.0);
	md_zsub2(data->N, data->out_dims, data->out_strs, data->tmp_dMss, data->map_strs, tmp_ones, data->out_strs, tmp_exp);

	// t*exp(-t.*scaling_R1s*R1s):
	md_zmul2(data->N, data->out_dims, data->out_strs, tmp_exp, data->out_strs, tmp_exp, data->TI_strs, multiplace_read(data->TI, dst));

	// scaling_R1s.*exp(-t.*scaling_R1s.*R1s).*t
	if (!use_compat_to_version("v0.6.00"))
		md_zsmul(data->N, data->out_dims, tmp_exp, tmp_exp, data->scaling_R1s);
	else
		md_zsmul(data->N, data->out_dims, tmp_exp, tmp_exp, data->scaling_M0);

	// scaling_M0.*M0
	md_zsmul2(data->N, data->map_dims, data->map_strs, tmp_map, data->map_strs, data->M0, data->scaling_M0);

	// Mss + scaling_M0*M0
	md_zadd(data->N, data->map_dims, tmp_ones, data->Mss, tmp_map);

	// R1s' = (Mss + scaling_M0*M0) * scaling_R1s.*exp(-t.*scaling_R1s.*R1s) * t
	md_zmul2(data->N, data->out_dims, data->out_strs, data->tmp_dR1s, data->map_strs, tmp_ones, data->out_strs, tmp_exp);

	md_free(tmp_map);
	md_free(tmp_ones);
	md_free(tmp_exp);
}

static void T1_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	struct T1_s* data = CAST_DOWN(T1_s, _data);
	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	// tmp = dM0
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, tmp_map, data->in_dims, src, CFL_SIZE);

	//const complex float* tmp_M0 = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = M0' * dM0
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, tmp_map, data->out_strs, data->tmp_dM0);

	// tmp = dMss
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, tmp_map, data->in_dims, src, CFL_SIZE);
	//const complex float* tmp_Mss = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);
	
	// dst = dst + dMss * Mss'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, tmp_map, data->out_strs, data->tmp_dMss);

	// tmp =  dR1s
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->map_dims, tmp_map, data->in_dims, src, CFL_SIZE);
	if (!use_compat_to_version("v0.6.00"))
		md_zreal(data->N, data->map_dims, tmp_map, tmp_map);

	//const complex float* tmp_R1s = (const void*)src + md_calc_offset(data->N, data->in_strs, pos);

	// dst = dst + dR1s * R1s'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, tmp_map, data->out_strs, data->tmp_dR1s);

	md_free(tmp_map);
}

static void T1_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	struct T1_s* data = CAST_DOWN(T1_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	// sum (conj(M0') * src, t)
	md_clear(data->N, data->map_dims,tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, tmp_map, data->out_strs, src, data->out_strs, data->tmp_dM0);

	// dst[1] = sum (conj(M0') * src, t)
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, tmp_map, CFL_SIZE);

	// sum (conj(Mss') * src, t)
	md_clear(data->N, data->map_dims, tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, tmp_map, data->out_strs, src, data->out_strs, data->tmp_dMss);

	// dst[0] = sum (conj(Mss') * src, t)
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, tmp_map, CFL_SIZE);

	// sum (conj(R1s') * src, t)
	md_clear(data->N, data->map_dims, tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, tmp_map, data->out_strs, src, data->out_strs, data->tmp_dR1s);

	if (!use_compat_to_version("v0.7.00"))
		md_zreal(data->N, data->map_dims, tmp_map, tmp_map);

	// dst[2] = sum (conj(R1s') * src, t)
	pos[COEFF_DIM] = 2;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, tmp_map, CFL_SIZE);

	md_free(tmp_map);
}

static void T1_del(const nlop_data_t* _data)
{
	struct T1_s* data = CAST_DOWN(T1_s, _data);

	md_free(data->Mss);
	md_free(data->M0);
	md_free(data->R1s);

	multiplace_free(data->TI);

	md_free(data->tmp_dM0);
	md_free(data->tmp_dMss);
	md_free(data->tmp_dR1s);

	xfree(data->map_dims);
	xfree(data->TI_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->map_strs);
	xfree(data->TI_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	xfree(data);
}


struct nlop_s* nlop_T1_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TI_dims[N], const complex float* TI, 
				float scaling_M0, float scaling_R1s)
{
	PTR_ALLOC(struct T1_s, data);
	SET_TYPEID(T1_s, data);


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
	data->Mss = NULL;
	data->M0 = NULL;
	data->R1s = NULL;
	data->tmp_dM0 = NULL;
	data->tmp_dMss = NULL;
	data->tmp_dR1s = NULL;

	data->TI = multiplace_move(N, TI_dims, CFL_SIZE, TI);

	data->scaling_M0 = scaling_M0;
	data->scaling_R1s = scaling_R1s;

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T1_fun, T1_der, T1_adj, NULL, NULL, T1_del);
}

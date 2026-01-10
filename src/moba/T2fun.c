/* Copyright 2018-2020. Uecker Lab, University Medical Center Goettingen.
 * Copyright 2022-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2019 Martin Uecker
 * 2018-2020 Xiaoqing Wang
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/multiplace.h"

#include "nlops/nlop.h"

#include "T2fun.h"



struct T2_s {

	nlop_data_t super;

	int N;

	const long* map_dims;
	const long* TE_dims;
	const long* in_dims;
	const long* out_dims;

	const long* map_strs;
	const long* TE_strs;
	const long* in_strs;
	const long* out_strs;

	// Parameter maps
	complex float* rho;
	complex float* z;

	complex float* drho;
	complex float* dz;

	struct multiplace_array_s* TE;
};

DEF_TYPEID(T2_s);

// Calculate Model: rho .*exp(-z.*TE)
static void T2_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);

	if (NULL == data->rho) {

		data->rho = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->z = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);
		data->drho = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
		data->dz = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
	}

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	// rho
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, data->rho, data->in_dims, src, CFL_SIZE);

	// z
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, data->z, data->in_dims, src, CFL_SIZE);

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	// -1*z
	md_zsmul(data->N, data->map_dims, tmp_map, data->z, -1.);

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
	// exp(-TE.*z)
	md_zmul2(data->N, data->out_dims, data->out_strs, tmp_exp, data->map_strs, tmp_map, data->TE_strs, multiplace_read(data->TE, dst));

	md_free(tmp_map);

	md_zexp(data->N, data->out_dims, tmp_exp, tmp_exp);

	// Calculating derivatives
	// drho
	md_zsmul(data->N, data->out_dims, data->drho, tmp_exp, 1.0);

	// model:
	// rho.*exp(-TE.*z)
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->rho, data->out_strs, tmp_exp);

	// dz: z' = -rho.*TE.*exp(-TE.*z)
	// TE.*exp(-TE.*z),
	md_zmul2(data->N, data->out_dims, data->out_strs, tmp_exp, data->out_strs, tmp_exp, data->TE_strs, multiplace_read(data->TE, dst));
	md_zsmul(data->N, data->out_dims, tmp_exp, tmp_exp, -1.);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dz, data->map_strs, data->rho, data->out_strs, tmp_exp);

	md_free(tmp_exp);
}

static void T2_der(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	// tmp = drho
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->map_dims, tmp_map, data->in_dims, src, CFL_SIZE);

	// dst = rho' * drho
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, tmp_map, data->out_strs, data->drho);

	// tmp =  dz
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->map_dims, tmp_map, data->in_dims, src, CFL_SIZE);

	// dst = dst + dz * z'
	md_zfmac2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, tmp_map, data->out_strs, data->dz);

	md_free(tmp_map);
}

static void T2_adj(const nlop_data_t* _data, int /*o*/, int /*i*/, complex float* dst, const complex float* src)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);

	long pos[data->N];

	for (int i = 0; i < data->N; i++)
		pos[i] = 0;

	complex float* tmp_map = md_alloc_sameplace(data->N, data->map_dims, CFL_SIZE, dst);

	// sum (conj(rho') * src, t)
	md_clear(data->N, data->map_dims, tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, tmp_map, data->out_strs, src, data->out_strs, data->drho);

	// dst[0] = sum (conj(rho') * src, t)
	pos[COEFF_DIM] = 0;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, tmp_map, CFL_SIZE);

	// sum (conj(z') * src, t)
	md_clear(data->N, data->map_dims, tmp_map, CFL_SIZE);
	md_zfmacc2(data->N, data->out_dims, data->map_strs, tmp_map, data->out_strs, src, data->out_strs, data->dz);
	// md_zreal(data->N, data->map_dims, data->tmp_map, data->tmp_map);

	// dst[1] = sum (conj(z') * src, t)
	pos[COEFF_DIM] = 1;
	md_copy_block(data->N, pos, data->in_dims, dst, data->map_dims, tmp_map, CFL_SIZE);

	md_free(tmp_map);
}

static void T2_del(const nlop_data_t* _data)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);

	md_free(data->rho);
	md_free(data->z);

	multiplace_free(data->TE);

	md_free(data->drho);
	md_free(data->dz);

	xfree(data->map_dims);
	xfree(data->TE_dims);
	xfree(data->in_dims);
	xfree(data->out_dims);

	xfree(data->map_strs);
	xfree(data->TE_strs);
	xfree(data->in_strs);
	xfree(data->out_strs);

	xfree(data);
}


struct nlop_s* nlop_T2_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TE_dims[N], const complex float* TE)
{
	PTR_ALLOC(struct T2_s, data);
	SET_TYPEID(T2_s, data);


	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, map_dims);
	data->map_dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nodims);
	md_copy_dims(N, *nodims, out_dims);
	data->out_dims = *PTR_PASS(nodims);

	PTR_ALLOC(long[N], nidims);
	md_copy_dims(N, *nidims, in_dims);
	data->in_dims = *PTR_PASS(nidims);

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

	PTR_ALLOC(long[N], ntestr);
	md_calc_strides(N, *ntestr, TE_dims, CFL_SIZE);
	data->TE_strs = *PTR_PASS(ntestr);

	data->N = N;
	data->rho = NULL;
	data->z = NULL;
	data->drho = NULL;
	data->dz = NULL;
	data->TE = multiplace_move(N, TE_dims, CFL_SIZE, TE);

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T2_fun, T2_der, T2_adj, NULL, NULL, T2_del);
}

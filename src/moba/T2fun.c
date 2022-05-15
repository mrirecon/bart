/* Copyright 2018-2019. Uecker Lab, University Medical Center Goettingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2020 Xiaoqing Wang <xiaoqing.wang@med.uni-goettingen.de>
 */


#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"

#include "T2fun.h"



struct T2_s {

	INTERFACE(nlop_data_t);

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

	complex float* TE;

	float scaling_z;
};

DEF_TYPEID(T2_s);

// Calculate Model: rho .*exp(-scaling_z.*z.*TE)
static void T2_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct T2_s* data = CAST_DOWN(T2_s, _data);
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

	// -1*scaling_z.*z
	md_zsmul(data->N, data->map_dims, tmp_map, data->z, -1. * data->scaling_z);

	complex float* tmp_exp = md_alloc_sameplace(data->N, data->out_dims, CFL_SIZE, dst);
	// exp(-TE.*scaling_z.*z)
	md_zmul2(data->N, data->out_dims, data->out_strs, tmp_exp, data->map_strs, tmp_map, data->TE_strs, data->TE);

	md_free(tmp_map);

	md_zexp(data->N, data->out_dims, tmp_exp, tmp_exp);

	// Calculating derivatives
	// drho
	md_zsmul(data->N, data->out_dims, data->drho, tmp_exp, 1.0);

	// model:
	// rho.*exp(-TE.*scaling_z.*z)
	md_zmul2(data->N, data->out_dims, data->out_strs, dst, data->map_strs, data->rho, data->out_strs, tmp_exp);

	// dz: z' = -rho.*scaling_z.*TE.*exp(-TE.*scaling_z.*z)
	// TE.*exp(-TE.*scaling_z.*z), 
	md_zmul2(data->N, data->out_dims, data->out_strs, tmp_exp, data->out_strs, tmp_exp, data->TE_strs, data->TE);
	md_zsmul(data->N, data->out_dims, tmp_exp, tmp_exp, -1. * data->scaling_z);
	md_zmul2(data->N, data->out_dims, data->out_strs, data->dz, data->map_strs, data->rho, data->out_strs, tmp_exp);

	md_free(tmp_exp);
}

static void T2_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

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

static void T2_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

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

	md_free(data->TE);

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


struct nlop_s* nlop_T2_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N], const long TE_dims[N], const complex float* TE, bool use_gpu)
{
#ifdef USE_CUDA
	md_alloc_fun_t my_alloc = use_gpu ? md_alloc_gpu : md_alloc;
#else
	assert(!use_gpu);
	md_alloc_fun_t my_alloc = md_alloc;
#endif

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
	data->rho = my_alloc(N, map_dims, CFL_SIZE);
	data->z = my_alloc(N, map_dims, CFL_SIZE);
	data->drho = my_alloc(N, out_dims, CFL_SIZE);
	data->dz = my_alloc(N, out_dims, CFL_SIZE);
	data->TE = my_alloc(N, TE_dims, CFL_SIZE);

	md_copy(N, TE_dims, data->TE, TE, CFL_SIZE);

	data->scaling_z = 10.;

	return nlop_create(N, out_dims, N, in_dims, CAST_UP(PTR_PASS(data)), T2_fun, T2_der, T2_adj, NULL, NULL, T2_del);
}

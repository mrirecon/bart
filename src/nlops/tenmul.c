/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "num/ops.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "tenmul.h"


struct tenmul_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;
	const long* dims1;
	const long* dims2;
	const long* ostr;
	const long* istr1;
	const long* istr2;
};

DEF_TYPEID(tenmul_s);

static void tenmul_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(tenmul_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

	nlop_data_der_alloc_memory(_data, dst);
	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	complex float* x1 = der_data[0];
	complex float* x2 = der_data[1];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif

	bool der1 = nlop_der_requested(_data, 0, 0);
	bool der2 = nlop_der_requested(_data, 1, 0);

	if (der2)
		md_copy2(data->N, data->dims1, MD_STRIDES(data->N, data->dims1, CFL_SIZE), x1, data->istr1, src1, CFL_SIZE);
	if (der1)
		md_copy2(data->N, data->dims2, MD_STRIDES(data->N, data->dims2, CFL_SIZE), x2, data->istr2, src2, CFL_SIZE);

	md_ztenmul2(data->N, data->dims, data->ostr, dst, data->istr1, src1, data->istr2, src2);
}

static void tenmul_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	complex float* x1 = der_data[0];
	//complex float* x2 = der_data[1];

	const auto data = CAST_DOWN(tenmul_s, _data);

	if (NULL == x1)
		error("Tenmul %x derivative not available\n", data);

	md_ztenmul2(data->N, data->dims, data->ostr, dst, data->istr2, src, MD_STRIDES(data->N, data->dims1, CFL_SIZE), x1);
}

static void tenmul_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	complex float* x1 = der_data[0];
	//complex float* x2 = der_data[1];

	const auto data = CAST_DOWN(tenmul_s, _data);

	if (NULL == x1)
		error("Tenmul %x derivative not available\n", data);


	md_ztenmulc2(data->N, data->dims, data->istr2, dst, data->ostr, src, MD_STRIDES(data->N, data->dims1, CFL_SIZE), x1);
}

static void tenmul_der1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	//complex float* x1 = der_data[0];
	complex float* x2 = der_data[1];

	const auto data = CAST_DOWN(tenmul_s, _data);

	if (NULL == x2)
		error("Tenmul %x derivative not available\n", data);

	md_ztenmul2(data->N, data->dims, data->ostr, dst, data->istr1, src, MD_STRIDES(data->N, data->dims2, CFL_SIZE), x2);
}

static void tenmul_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	void* der_data[2];
	nlop_get_der_array(_data, 2, (void**)der_data);
	//complex float* x1 = der_data[0];
	complex float* x2 = der_data[1];

	const auto data = CAST_DOWN(tenmul_s, _data);

	if (NULL == x2)
		error("Tenmul %x derivative not available\n", data);

	md_ztenmulc2(data->N, data->dims, data->istr1, dst, data->ostr, src, MD_STRIDES(data->N, data->dims2, CFL_SIZE), x2);
}


static void tenmul_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(tenmul_s, _data);

	xfree(data->dims);
	xfree(data->ostr);
	xfree(data->dims1);
	xfree(data->istr1);
	xfree(data->dims2);
	xfree(data->istr2);
	xfree(data);
}


struct nlop_s* nlop_tenmul_create2(int N, const long dims[N], const long ostr[N],
		const long istr1[N], const long istr2[N])
{

	PTR_ALLOC(struct tenmul_s, data);
	SET_TYPEID(tenmul_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	PTR_ALLOC(long[N], nostr);
	md_copy_strides(N, *nostr, ostr);

	PTR_ALLOC(long[N], ndims1);
	PTR_ALLOC(long[N], nistr1);
	md_select_dims(N, md_nontriv_strides(N, istr1), *ndims1, dims);
	md_copy_strides(N, *nistr1, istr1);

	PTR_ALLOC(long[N], ndims2);
	PTR_ALLOC(long[N], nistr2);
	md_select_dims(N, md_nontriv_strides(N, istr2), *ndims2, dims);
	md_copy_strides(N, *nistr2, istr2);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->ostr = *PTR_PASS(nostr);
	data->dims1 = *PTR_PASS(ndims1);
	data->istr1 = *PTR_PASS(nistr1);
	data->dims2 = *PTR_PASS(ndims2);
	data->istr2 = *PTR_PASS(nistr2);

	long nl_odims[1][N];
	md_select_dims(N, md_nontriv_strides(N, ostr), nl_odims[0], dims);

	long nl_ostr[1][N];
	md_copy_strides(N, nl_ostr[0], ostr);

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], data->dims1);
	md_copy_dims(N, nl_idims[1], data->dims2);

	long nl_istr[2][N];
	md_copy_strides(N, nl_istr[0], istr1);
	md_copy_strides(N, nl_istr[1], istr2);

	struct nlop_der_array_s* tenmul_der_arrays[2];
	tenmul_der_arrays[0] = nlop_der_array_create(N, nl_idims[0], CFL_SIZE, 2, 1, (bool[2][1]){{ false }, { true }});
	tenmul_der_arrays[1] = nlop_der_array_create(N, nl_idims[1], CFL_SIZE, 2, 1, (bool[2][1]){{ true }, { false }});

	return nlop_generic_managed_create2(1, N, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)),
		tenmul_fun, (nlop_der_fun_t[2][1]){ { tenmul_der1 }, { tenmul_der2 } }, (nlop_der_fun_t[2][1]){ { tenmul_adj1 }, { tenmul_adj2 } }, NULL, NULL, tenmul_del, 2, tenmul_der_arrays, NULL);
}


struct nlop_s* nlop_tenmul_create(int N, const long odim[N], const long idim1[N], const long idim2[N])
{
	long dims[N];
	md_tenmul_dims(N, dims, odim, idim1, idim2);

	return nlop_tenmul_create2(N, dims, MD_STRIDES(N, odim, CFL_SIZE),
					MD_STRIDES(N, idim1, CFL_SIZE),
					MD_STRIDES(N, idim2, CFL_SIZE));
}

bool nlop_tenmul_der_available(const struct nlop_s* op, int index)
{
	auto data = CAST_MAYBE(tenmul_s, nlop_get_data((struct nlop_s*)op));
	assert(NULL != data);

	void* der_data[2];
	nlop_get_der_array(nlop_get_data((struct nlop_s*)op), 2, (void**)der_data);
	complex float* x1 = der_data[0];
	complex float* x2 = der_data[1];

	if (0 == index)
		return (NULL != x2);
	if (1 == index)
		return (NULL != x1);
	assert(0);
	return false;
}

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
#include "nlops/nlop_jacobian.h"
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

	complex float* der1;
	complex float* der2;
};

DEF_TYPEID(tenmul_s);

static void tenmul_init(struct tenmul_s* data, const complex float* ref)
{
	if (nlop_der_requested(CAST_UP(data), 0, 0)) {

		if (NULL == data->der2)
			data->der2 = md_alloc_sameplace(data->N, data->dims2, CFL_SIZE, ref);
	} else {

		md_free(data->der2);
		data->der2 = NULL;
	}

	if (nlop_der_requested(CAST_UP(data), 1, 0)) {

		if (NULL == data->der1)
			data->der1 = md_alloc_sameplace(data->N, data->dims1, CFL_SIZE, ref);
	} else {

		md_free(data->der1);
		data->der1 = NULL;
	}
}

static void tenmul_clear_der(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(tenmul_s, _data);

	md_free(data->der1);
	md_free(data->der2);

	data->der1 = NULL;
	data->der2 = NULL;
}

static void tenmul_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(tenmul_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

	tenmul_init(data, dst);

	complex float* x1 = data->der1;
	complex float* x2 = data->der2;

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif

	if (nlop_der_requested(_data, 1, 0))
		md_copy2(data->N, data->dims1, MD_STRIDES(data->N, data->dims1, CFL_SIZE), x1, data->istr1, src1, CFL_SIZE);
	if (nlop_der_requested(_data, 0, 0))
		md_copy2(data->N, data->dims2, MD_STRIDES(data->N, data->dims2, CFL_SIZE), x2, data->istr2, src2, CFL_SIZE);

	md_ztenmul2(data->N, data->dims, data->ostr, dst, data->istr1, src1, data->istr2, src2);
}

static void tenmul_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(tenmul_s, _data);
	complex float* x1 = data->der1;

	if (NULL == x1)
		error("Tenmul %x derivative not available\n", data);

	md_ztenmul2(data->N, data->dims, data->ostr, dst, data->istr2, src, MD_STRIDES(data->N, data->dims1, CFL_SIZE), x1);
}

static void tenmul_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(tenmul_s, _data);
	complex float* x1 = data->der1;

	if (NULL == x1)
		error("Tenmul %x derivative not available\n", data);


	md_ztenmulc2(data->N, data->dims, data->istr2, dst, data->ostr, src, MD_STRIDES(data->N, data->dims1, CFL_SIZE), x1);
}

static void tenmul_der1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(tenmul_s, _data);
	complex float* x2 = data->der2;

	if (NULL == x2)
		error("Tenmul %x derivative not available\n", data);

	md_ztenmul2(data->N, data->dims, data->ostr, dst, data->istr1, src, MD_STRIDES(data->N, data->dims2, CFL_SIZE), x2);
}

static void tenmul_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(tenmul_s, _data);
	complex float* x2 = data->der2;

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

	md_free(data->der1);
	md_free(data->der2);

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

	data->der1 = NULL;
	data->der2 = NULL;

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

	return nlop_generic_managed_create2(1, N, nl_odims, nl_ostr, 2, N, nl_idims, nl_istr, CAST_UP(PTR_PASS(data)),
		tenmul_fun, (nlop_der_fun_t[2][1]){ { tenmul_der1 }, { tenmul_der2 } }, (nlop_der_fun_t[2][1]){ { tenmul_adj1 }, { tenmul_adj2 } }, NULL, NULL, tenmul_del, tenmul_clear_der, NULL);
}

struct tenmul_block_diag_s {

	INTERFACE(nlop_data_t);
};

DEF_TYPEID(tenmul_block_diag_s);

static void tenmul_block_diag_fun(const nlop_data_t* _data, int N, int OO, const long odims[OO][N], _Complex float* dst[OO], int II, const long idims[II][N], const _Complex float* src[II], const long ddims[OO][II][N], _Complex float* jac[OO][II])
{
	UNUSED(_data);

	assert(1 == OO);
	assert(2 == II);

	assert(md_check_equal_dims(N, idims[0], ddims[0][1], ~0));
	assert(md_check_equal_dims(N, idims[1], ddims[0][0], ~0));

	md_ztenmul(N, odims[0], dst[0], idims[0], src[0], idims[1], src[1]);

	if (NULL != jac[0][0])
		md_copy(N, ddims[0][0], jac[0][0], src[1], CFL_SIZE);

	if (NULL != jac[0][1])
		md_copy(N, ddims[0][1], jac[0][1], src[0], CFL_SIZE);
}

struct nlop_s* nlop_tenmul_create(int N, const long odim[N], const long idim1[N], const long idim2[N])
{
	PTR_ALLOC(struct tenmul_block_diag_s, data);
	SET_TYPEID(tenmul_block_diag_s, data);

	int OO = 1;
	int II = 2;

	long odims[OO][N];
	long idims[II][N];

	unsigned long diag_flags[OO][II];

	md_copy_dims(N, odims[0], odim);
	md_copy_dims(N, idims[0], idim1);
	md_copy_dims(N, idims[1], idim2);

	diag_flags[0][0] = ~md_nontriv_dims(N, idim2);
	diag_flags[0][1] = ~md_nontriv_dims(N, idim1);

	return nlop_zblock_diag_generic_create(CAST_UP(PTR_PASS(data)), N, OO, odims, II, idims, diag_flags, tenmul_block_diag_fun, NULL);
}

bool nlop_tenmul_der_available(const struct nlop_s* op, int index)
{
	auto data = CAST_MAYBE(tenmul_s, nlop_get_data((struct nlop_s*)op));
	if (NULL != data) {

		if (0 == index)
			return (NULL != data->der2);
		if (1 == index)
			return (NULL != data->der1);
		assert(0);
	}

	return nlop_block_diag_der_available(op, 0, index);
}

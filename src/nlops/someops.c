/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "linops/linop.h"
#include "linops/someops.h"
#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/iovec.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/tenmul.h"
#include "nlops/nlop_jacobian.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "someops.h"


struct zaxpbz_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;

	const long* ostrs;
	const long* istrs1;
	const long* istrs2;

	complex float scale1;
	complex float scale2;
};

DEF_TYPEID(zaxpbz_s);

static void zaxpbz_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(zaxpbz_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	complex float* src1 = args[1];
	complex float* src2 = args[2];

	#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
	#endif

	src1 = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, args[1]);
	src2 = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, args[2]);

	md_copy2(data->N, data->dims, data->ostrs, src1, data->istrs1, args[1], CFL_SIZE);
	md_copy2(data->N, data->dims, data->ostrs, src2, data->istrs2, args[2], CFL_SIZE);

	const long* istrs1 = data->ostrs;
	const long* istrs2 = data->ostrs;



	if ((1. == data->scale1) && (1. == data->scale2)) {

		md_zadd2(data->N, data->dims, data->ostrs, dst, istrs1, src1, istrs2, src2);
		goto cleanup;
	}

	if ((1. == data->scale1) && (-1. == data->scale2)) {

		md_zsub2(data->N, data->dims, data->ostrs, dst, istrs1, src1, istrs2, src2);
		goto cleanup;
	}

	if ((-1. == data->scale1) && (1. == data->scale2)) {

		md_zsub2(data->N, data->dims, data->ostrs, dst, istrs2, src2, istrs1, src1);
		goto cleanup;
	}

	md_zsmul2(data->N, data->dims, data->ostrs, dst, istrs1, src1, data->scale1);
	md_zaxpy2(data->N, data->dims, data->ostrs, dst, data->scale2, istrs2, src2);

cleanup:

	if (args[1] != src1)
		md_free(src1);

	if (args[2] != src2)
		md_free(src2);
}

static void scale_apply(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	const auto data = CAST_DOWN(zaxpbz_s, _data);
	const complex float scale = (i == 0) ? data->scale1 : data->scale2;

	md_zsmul2(data->N, data->dims, data->ostrs, dst, (i == 0) ? data->istrs1 : data->istrs2, src, scale);
}

static void scale_adjoint(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	const auto data = CAST_DOWN(zaxpbz_s, _data);
	const complex float scale = (i == 0) ? data->scale1 : data->scale2;

	md_clear2(data->N, data->dims, (i == 0) ? data->istrs1 : data->istrs2, dst, CFL_SIZE); //FIXME: compute size and use md_clear
	md_zaxpy2(data->N, data->dims, (i == 0) ? data->istrs1 : data->istrs2, dst, conjf(scale), data->ostrs, src);
}


static void zaxpbz_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zaxpbz_s, _data);

	xfree(data->dims);
	xfree(data->ostrs);
	xfree(data->istrs1);
	xfree(data->istrs2);

	xfree(data);
}

const struct nlop_s* nlop_zaxpbz2_create(int N, const long dims[N], unsigned long flags1, complex float scale1, unsigned long flags2, complex float scale2)
{
	PTR_ALLOC(struct zaxpbz_s, data);
	SET_TYPEID(zaxpbz_s, data);

	PTR_ALLOC(long[N], ndims);
	PTR_ALLOC(long[N], ostrs);
	PTR_ALLOC(long[N], istrs1);
	PTR_ALLOC(long[N], istrs2);

	md_copy_dims(N, *ndims, dims);
	md_calc_strides(N, *ostrs, dims, CFL_SIZE);

	long idims1[N];
	long idims2[N];

	md_select_dims(N, flags1, idims1, dims);
	md_select_dims(N, flags2, idims2, dims);

	md_calc_strides(N, *istrs1, idims1, CFL_SIZE);
	md_calc_strides(N, *istrs2, idims2, CFL_SIZE);

	data->N = N;
	data->dims = *PTR_PASS(ndims);
	data->ostrs = *PTR_PASS(ostrs);
	data->istrs1 = *PTR_PASS(istrs1);
	data->istrs2 = *PTR_PASS(istrs2);

	data->scale1 = scale1;
	data->scale2 = scale2;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);


	long nl_idims[2][N];
	md_select_dims(N, flags1, nl_idims[0], dims);
	md_select_dims(N, flags2, nl_idims[1], dims);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
		zaxpbz_fun, (nlop_der_fun_t[2][1]){ { scale_apply }, { scale_apply } }, (nlop_der_fun_t[2][1]){ { scale_adjoint }, { scale_adjoint } }, NULL, NULL, zaxpbz_del);
}

const struct nlop_s* nlop_zaxpbz_create(int N, const long dims[N], complex float scale1, complex float scale2)
{
	return nlop_zaxpbz2_create(N, dims, ~0, scale1, ~0, scale2);
}

const struct nlop_s* nlop_zsadd_create(int N, const long dims[N], complex float val)
{
	auto result = nlop_zaxpbz_create(N, dims, 1., 1.);
	return nlop_set_input_const_F2(result, 1, N, dims, MD_SINGLETON_STRS(N), true, &val);
}


struct dump_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const long* dims;

	const char* filename;
	long counter;

	bool frw;
	bool der;
	bool adj;
};

DEF_TYPEID(dump_s);

static void dump_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(dump_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);

	if (data->frw) {

		char filename[strlen(data->filename) + 10];
		sprintf(filename, "%s_%ld_frw", data->filename, data->counter);
		dump_cfl(filename, data->N, data->dims, src);
		data->counter++;
	}
}

static void dump_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(dump_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);

	if (data->der) {

		char filename[strlen(data->filename) + 10];
		sprintf(filename, "%s_%ld_der", data->filename, data->counter);
		dump_cfl(filename, data->N, data->dims, src);
		data->counter++;
	}
}

static void dump_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(dump_s, _data);

	md_copy(data->N, data->dims, dst, src, CFL_SIZE);

	if (data->adj) {

		char filename[strlen(data->filename) + 10];
		sprintf(filename, "%s_%ld_adj", data->filename, data->counter);
		dump_cfl(filename, data->N, data->dims, src);
		data->counter++;
	}
}

static void dump_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(dump_s, _data);

	xfree(data->dims);
	xfree(data);
}

/**
 * Operator dumping its input to a filename_%d_frw/der/adj.cfl file
 * @param N
 * @param dims
 * @param filename
 * @param frw - store frw input
 * @param der - store der input
 * @param adj - store adj input
 */

const struct nlop_s* nlop_dump_create(int N, const long dims[N], const char* filename, bool frw, bool der, bool adj)
{
	PTR_ALLOC(struct dump_s, data);
	SET_TYPEID(dump_s, data);

	data->N = N;

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);

	PTR_ALLOC(char[strlen(filename) + 1], nfilename);
	strcpy(*nfilename, filename);
	data->filename = *PTR_PASS(nfilename);

	data->frw = frw;
	data->der = der;
	data->adj = adj;

	data->counter = 0;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), dump_fun, dump_der, dump_adj, NULL, NULL, dump_del);
}


struct zinv_reg_s {

	INTERFACE(nlop_data_t);

	complex float epsilon;
};

DEF_TYPEID(zinv_reg_s);

static void zinv_reg_fun(const nlop_data_t* _data, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{
	const auto data = CAST_DOWN(zinv_reg_s, _data);

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);

	md_zfill(N, dims, tmp, 1.);
	md_zdiv_reg(N, dims, dst, tmp, src, data->epsilon);

	md_free(tmp);

	if (NULL != der) {

		md_zmul(N, dims, der, dst, dst);
		md_zsmul(N, dims, der, der, -1.);
	}
}

/**
 * Operator computing the inverse
 * f(x) = 1 / (x + eps)
 */
const struct nlop_s* nlop_zinv_reg_create(int N, const long dims[N], float eps)
{
	PTR_ALLOC(struct zinv_reg_s, data);
	SET_TYPEID(zinv_reg_s, data);

	data->epsilon = eps;

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zinv_reg_fun, NULL);
}

/**
 * Operator computing the inverse
 * f(x) = 1 / (x)
 */
const struct nlop_s* nlop_zinv_create(int N, const long dims[N])
{
	return nlop_zinv_reg_create(N, dims, 0);
}

/**
 * Operator dividing input one by input 2
 * f(x, y) = x / (y + eps)
 */
const struct nlop_s* nlop_zdiv_reg_create(int N, const long dims[N], float eps)
{
	return nlop_chain2_FF(nlop_zinv_reg_create(N, dims, eps), 0, nlop_tenmul_create(N, dims, dims, dims), 1);
}

/**
 * Operator dividing input one by input 2
 * f(x, y) = x / y
 */
const struct nlop_s* nlop_zdiv_create(int N, const long dims[N])
{
	return nlop_chain2_FF(nlop_zinv_create(N, dims), 0, nlop_tenmul_create(N, dims, dims, dims), 1);
}

struct zmax_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	unsigned long flags;
	const long* outdims;
	const long* dims;
	const long* strides;
	const long* outstrides;

	complex float* max_index;
};

DEF_TYPEID(zmax_s);

static void zmax_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zmax_s, _data);

	if (NULL == data->max_index)
		data->max_index = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	md_copy2(data->N, data->outdims, data->outstrides, dst, data->strides, src, CFL_SIZE);

	md_zmax2(data->N, data->dims, data->outstrides, dst, data->outstrides, dst, data->strides, src);

#ifdef USE_CUDA
	if (cuda_ondevice(dst)) {

		md_copy2(data->N, data->dims, data->strides, data->max_index, data->outstrides, dst, CFL_SIZE);
		md_zgreatequal(data->N, data->dims, data->max_index, src, data->max_index);
	} else
#endif
	md_zgreatequal2(data->N, data->dims, data->strides, data->max_index, data->strides, src, data->outstrides, dst);
}

static void zmax_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zmax_s, _data);

	md_ztenmul(data->N, data->outdims, dst, data->dims, src, data->dims, data->max_index);
	md_zreal(data->N, data->outdims, dst, dst);
}

static void zmax_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(zmax_s, _data);

	md_zmul2(data->N, data->dims, data->strides, dst, data->outstrides, src, data->strides, data->max_index);
	md_zreal(data->N, data->dims, dst, dst);
}

static void zmax_del(const struct nlop_data_s* _data)
{
	const auto data = CAST_DOWN(zmax_s, _data);

	md_free(data->max_index);

	xfree(data->outdims);
	xfree(data->dims);
	xfree(data->strides);
	xfree(data->outstrides);

	xfree(data);
}


/**
 * Returns maximum value of array along specified flags.
 **/
const struct nlop_s* nlop_zmax_create(int N, const long dims[N], unsigned long flags)
{
	PTR_ALLOC(struct zmax_s, data);
	SET_TYPEID(zmax_s, data);

	PTR_ALLOC(long[N], outdims);
	md_select_dims(N, ~flags, *outdims, dims);

	PTR_ALLOC(long[N], dims_tmp);
	md_copy_dims(N, *dims_tmp, dims);

	PTR_ALLOC(long[N], strides);
	md_calc_strides(N, *strides, dims, CFL_SIZE);

	PTR_ALLOC(long[N], out_strides);
	md_calc_strides(N, *out_strides, *outdims, CFL_SIZE);

	data->N = N;
	data->flags = flags;
	data->strides = *PTR_PASS(strides);
	data->dims = *PTR_PASS(dims_tmp);
	data->outdims = *PTR_PASS(outdims);
	data->outstrides = *PTR_PASS(out_strides);

	data->max_index = NULL;

	long odims[N];
	md_select_dims(N, ~flags, odims, dims);

	return nlop_create(N, odims, N, dims, CAST_UP(PTR_PASS(data)), zmax_fun, zmax_der, zmax_adj, NULL, NULL, zmax_del);
}



struct zsqrt_s { INTERFACE(nlop_data_t); };

DEF_TYPEID(zsqrt_s);

static void zsqrt_free(const nlop_data_t* _data)
{
	xfree(_data);
}

static void zsqrt_apply(const nlop_data_t* _data, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{
	UNUSED(_data);

	md_zsqrt(N, dims, dst, src);

	if (NULL != der) {

		md_zfill(N, dims, der, 0.5);
		md_zdiv(N, dims, der, der, dst);
	}
}

const struct nlop_s* nlop_zsqrt_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zsqrt_s, data);
	SET_TYPEID(zsqrt_s, data);

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zsqrt_apply, zsqrt_free);
}



/**
 * Returns zss of array along specified flags.
 **/
const struct nlop_s* nlop_zss_create(int N, const long dims[N], unsigned long flags)
{
	long odims[N];
	md_select_dims(N, ~flags, odims, dims);

	auto result = nlop_tenmul_create(N, odims, dims, dims);

	result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, dims)), 0, result, 0);
	result = nlop_dup_F(result, 0, 1);

	result = nlop_chain_FF(result, nlop_from_linop_F(linop_zreal_create(N, odims)));

	return result;
}

/**
 * Returns zrss of array along specified flags.
 **/
const struct nlop_s* nlop_zrss_reg_create(int N, const long dims[N], unsigned long flags, float epsilon)
{
	long odims[N];
	md_select_dims(N, ~flags, odims, dims);

	auto result = nlop_zss_create(N, dims, flags);

	if (0 != epsilon)
		result = nlop_chain_FF(result, nlop_zsadd_create(N, odims, epsilon));

	result = nlop_chain_FF(result, nlop_zsqrt_create(N, odims));
	result = nlop_chain_FF(result, nlop_from_linop_F(linop_zreal_create(N, odims)));

	return result;
}

const struct nlop_s* nlop_zrss_create(int N, const long dims[N], unsigned long flags)
{
	return nlop_zrss_reg_create(N, dims, flags, 0);
}



struct zspow_s {

	INTERFACE(nlop_data_t);
	complex float exp;
};

DEF_TYPEID(zspow_s);

static void zspow_fun(const nlop_data_t* _data, int N, const long dims[N], complex float* dst, const complex float* src, complex float* der)
{
	const auto data = CAST_DOWN(zspow_s, _data);

	md_zspow(N, dims, dst, src, data->exp);

	if (NULL != der) {

		md_zdiv(N, dims, der, dst, src);
		md_zsmul(N, dims, der, der, data->exp);
	}
}

/**
 * Operator computing the inverse
 * f(x) = x^p
 */
const struct nlop_s* nlop_zspow_create(int N, const long dims[N], complex float exp)
{
	PTR_ALLOC(struct zspow_s, data);
	SET_TYPEID(zspow_s, data);

	data->exp = exp;

	return nlop_zdiag_create(N, dims, CAST_UP(PTR_PASS(data)), zspow_fun, NULL);
}


/**
 * Operator computing the smoothed pointwise absolute value
 * f(x) = sqrt(re(x)^2 + im (x)^2 + epsilon)
 */
const struct nlop_s* nlop_smo_abs_create(int N, const long dims[N], float epsilon)
{
	return nlop_zrss_reg_create(N, dims, 0, epsilon);
}

const struct nlop_s* nlop_zabs_create(int N, const long dims[N])
{
	return nlop_zrss_reg_create(N, dims, 0, 0);
}


/**
 * Operator extracting unit-norm complex exponentials from complex arrays
 * f(x) = x / |x|
 */
const struct nlop_s* nlop_zphsr_create(int N, const long dims[N])
{
	auto result = nlop_zdiv_create(N, dims);
	result = nlop_chain2_FF(nlop_zabs_create(N, dims), 0, result, 1);
	result = nlop_dup_F(result, 0, 0);
	return result;
}

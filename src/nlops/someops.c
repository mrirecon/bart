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

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "someops.h"


struct zaxpbz_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* dims;

	complex float scale1;
	complex float scale2;
};

DEF_TYPEID(zaxpbz_s);

static void zaxpbz_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(zaxpbz_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

	#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
	#endif

	if ((1. == data->scale1) && (1. == data->scale2)) {

		md_zadd(data->N, data->dims, dst, src1, src2);
			return;
	}

	if ((1. == data->scale1) && (-1. == data->scale2)) {

		md_zsub(data->N, data->dims, dst, src1, src2);
			return;
	}

	if ((-1. == data->scale1) && (1. == data->scale2)) {

		md_zsub(data->N, data->dims, dst, src2, src1);
			return;
	}

	md_zsmul(data->N, data->dims, dst, src1, data->scale1);
	md_zaxpy(data->N, data->dims, dst, data->scale2, src2);
}

static void scale_apply(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	const auto data = CAST_DOWN(zaxpbz_s, _data);
	const complex float scale = (i == 0) ? data->scale1 : data->scale2;
	md_zsmul(data->N, data->dims, dst, src, scale);
}

static void scale_adjoint(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	assert(0 == o);
	const auto data = CAST_DOWN(zaxpbz_s, _data);
	const complex float scale = (i == 0) ? data->scale1 : data->scale2;
	md_zsmul(data->N, data->dims, dst, src, conjf(scale));
}


static void zaxpbz_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zaxpbz_s, _data);

	xfree(data->dims);

	xfree(data);
}

const struct nlop_s* nlop_zaxpbz_create(int N, const long dims[N], complex float scale1, complex float scale2)
{

	PTR_ALLOC(struct zaxpbz_s, data);
	SET_TYPEID(zaxpbz_s, data);

	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);

	data->N = N;
	data->dims = *PTR_PASS(ndims);

	data->scale1 = scale1;
	data->scale2 = scale2;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], dims);


	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], dims);
	md_copy_dims(N, nl_idims[1], dims);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
		zaxpbz_fun, (nlop_der_fun_t[2][1]){ { scale_apply }, { scale_apply } }, (nlop_der_fun_t[2][1]){ { scale_adjoint }, { scale_adjoint } }, NULL, NULL, zaxpbz_del);
}

struct smo_abs_s {

	INTERFACE(nlop_data_t);

	unsigned long N;
	const long* dims;

	complex float epsilon;
	complex float* tmp;
};

DEF_TYPEID(smo_abs_s);

static void smo_abs_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(smo_abs_s, _data);

	long rdims[data->N + 1];
	rdims[0] = 2;
	md_copy_dims(data->N, rdims + 1, data->dims);

	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(data->N, data->dims, CFL_SIZE, dst);

	md_zmulc(data->N, data->dims, dst, src, src);//dst=[r0^2 + i0^2 + 0i, r1^2 + i1^2 + 0i, ...]
	md_zreal(data->N, data->dims, dst, dst); //zmulc does not gurantee vanishing imag on gpu
	md_zsadd(data->N, data->dims, dst, dst, data->epsilon);
	md_sqrt(data->N+1, rdims, (float*)dst, (float*)dst);
	md_zdiv(data->N, data->dims, data->tmp, src, dst);

}


static void smo_abs_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct smo_abs_s* data = CAST_DOWN(smo_abs_s, _data);
	assert(NULL != data->tmp);

	md_zmulc(data->N, data->dims, dst, data->tmp, src);
	md_zreal(data->N, data->dims, dst, dst);
}

static void smo_abs_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);


	const struct smo_abs_s* data = CAST_DOWN(smo_abs_s, _data);
	assert(NULL != data->tmp);

	md_zreal(data->N, data->dims, dst, src);
	md_zmul(data->N, data->dims, dst, dst, data->tmp);

}

static void smo_abs_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(smo_abs_s, _data);

	md_free(data->tmp);
	xfree(data->dims);
	xfree(data);
}

/**
 * Operator computing the smoothed pointwise absolute value
 * f(x) = sqrt(re(x)^2 + im (x)^2 + epsilon)
 */
const struct nlop_s* nlop_smo_abs_create(int N, const long dims[N], float epsilon)
{
	PTR_ALLOC(struct smo_abs_s, data);
	SET_TYPEID(smo_abs_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->epsilon = epsilon;

	// will be initialized later, to transparently support GPU
	data->tmp = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), smo_abs_fun, smo_abs_der, smo_abs_adj, NULL, NULL, smo_abs_del);
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

	unsigned long N;
	const long* dims;

	complex float epsilon; //reg not implemented
	complex float* tmp;
};

DEF_TYPEID(zinv_reg_s);

static void zinv_reg_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(zinv_reg_s, _data);

	unsigned int N = data->N;
	const long* dims = data->dims;

	if (NULL == data->tmp)
		data->tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);

	md_zfill(N, dims, data->tmp, 1.);
	md_zdiv(N,dims, dst, data->tmp, src);

	md_zmul(N, dims, data->tmp, dst, dst);
	md_zsmul(N, dims, data->tmp, data->tmp, -1.);
}


static void zinv_reg_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct zinv_reg_s* data = CAST_DOWN(zinv_reg_s, _data);
	assert(NULL != data->tmp);

	md_zmul(data->N, data->dims, dst, src, data->tmp);
}

static void zinv_reg_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const struct zinv_reg_s* data = CAST_DOWN(zinv_reg_s, _data);
	assert(NULL != data->tmp);

	md_zmulc(data->N, data->dims, dst, src, data->tmp);
}

static void zinv_reg_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(zinv_reg_s, _data);

	md_free(data->tmp);
	xfree(data->dims);
	xfree(data);
}

/**
 * Operator computing the inverse
 * f(x) = 1 / x
 */
const struct nlop_s* nlop_zinv_create(int N, const long dims[N])
{
	PTR_ALLOC(struct zinv_reg_s, data);
	SET_TYPEID(zinv_reg_s, data);

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->epsilon = 0;

	// will be initialized later, to transparently support GPU
	data->tmp = NULL;

	return nlop_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), zinv_reg_fun, zinv_reg_der, zinv_reg_adj, NULL, NULL, zinv_reg_del);
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
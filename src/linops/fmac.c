/* Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <string.h>
#include <complex.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/multiplace.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "fmac.h"






struct fmac_data {

	INTERFACE(linop_data_t);

	unsigned int N;
	long *dims;

	long *idims;
	long *istrs;

	long *odims;
	long *ostrs;

	long *tdims;
	long *tstrs;

	struct multiplace_array_s* tensor;
};

static DEF_TYPEID(fmac_data);

static void fmac_free_data(const linop_data_t* _data)
{
	auto data = CAST_DOWN(fmac_data, _data);

	multiplace_free(data->tensor);

	xfree(data->dims);
	xfree(data->idims);
	xfree(data->istrs);
	xfree(data->odims);
	xfree(data->ostrs);
	xfree(data->tdims);
	xfree(data->tstrs);

	xfree(data);
}


static void fmac_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(fmac_data, _data);

	md_ztenmul2(data->N, data->dims, data->ostrs, dst, data->istrs, src, data->tstrs, multiplace_read(data->tensor, dst));
}

static void fmac_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(fmac_data, _data);

	md_ztenmulc2(data->N, data->dims, data->istrs, dst, data->ostrs, src, data->tstrs, multiplace_read(data->tensor, dst));
}

static void fmac_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	struct fmac_data* data = CAST_DOWN(fmac_data, _data);

	complex float* tmp = md_alloc_sameplace(data->N, data->odims, CFL_SIZE, dst);
	fmac_apply(_data, tmp, src);
	fmac_adjoint(_data, dst, tmp);
	md_free(tmp);
}

const struct linop_s* linop_fmac_create(unsigned int N, const long dims[N],
		unsigned int oflags, unsigned int iflags, unsigned int tflags, const complex float* tensor)
{
	PTR_ALLOC(struct fmac_data, data);
	SET_TYPEID(fmac_data, data);

	data->N = N;

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);

	data->idims = *TYPE_ALLOC(long[N]);
	data->istrs = *TYPE_ALLOC(long[N]);

	md_select_dims(N, ~iflags, data->idims, dims);
	md_calc_strides(N, data->istrs, data->idims, CFL_SIZE);

	data->odims = *TYPE_ALLOC(long[N]);
	data->ostrs = *TYPE_ALLOC(long[N]);

	md_select_dims(N, ~oflags, data->odims, dims);
	md_calc_strides(N, data->ostrs, data->odims, CFL_SIZE);

	data->tstrs = *TYPE_ALLOC(long[N]);
	data->tdims = *TYPE_ALLOC(long[N]);

	md_select_dims(N, ~tflags, data->tdims, dims);
	md_calc_strides(N, data->tstrs, data->tdims, CFL_SIZE);

	data->tensor = (NULL == tensor) ? NULL : multiplace_move(N, data->tdims, CFL_SIZE, tensor);

	long odims[N];
	md_copy_dims(N, odims, data->odims);

	long idims[N];
	md_copy_dims(N, idims, data->idims);

	return linop_create(N, odims, N, idims,
			CAST_UP(PTR_PASS(data)), fmac_apply, fmac_adjoint, fmac_normal,
			NULL, fmac_free_data);
}

void linop_fmac_set_tensor(const struct linop_s* lop, int N, const long tdims[N], const complex float* tensor)
{
	auto _data = linop_get_data(lop);
	auto data = CAST_DOWN(fmac_data, _data);

	assert(data->N == (unsigned int)N);
	assert(md_check_equal_dims(N, tdims, data->tdims, ~0));

	multiplace_free(data->tensor);

	data->tensor = multiplace_move(N, data->tdims, CFL_SIZE, tensor);
}


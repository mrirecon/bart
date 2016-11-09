/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <string.h>
#include <complex.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
//#include "num/iovec.h"

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

	long *tstrs;

	const complex float* tensor;
#ifdef USE_CUDA
	const complex float* gpu_tensor;
#endif
};

DEF_TYPEID(fmac_data);



static void fmac_free_data(const linop_data_t* _data)
{
        struct fmac_data* data = CAST_DOWN(fmac_data, _data);

	xfree(data->dims);
	xfree(data->idims);
	xfree(data->istrs);
	xfree(data->odims);
	xfree(data->ostrs);
	xfree(data->tstrs);

	xfree(data);
}


static void fmac_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct fmac_data* data = CAST_DOWN(fmac_data, _data);

	// FIXME: gpuA

	md_clear2(data->N, data->odims, data->ostrs, dst, CFL_SIZE);
	md_zfmac2(data->N, data->dims, data->ostrs, dst, data->istrs, src, data->tstrs, data->tensor);
}

static void fmac_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        struct fmac_data* data = CAST_DOWN(fmac_data, _data);

	// FIXME: gpu

	md_clear2(data->N, data->idims, data->istrs, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->dims, data->istrs, dst, data->ostrs, src, data->tstrs, data->tensor);
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

	long tdims[N];
	md_select_dims(N, ~tflags, tdims, dims);
	md_calc_strides(N, data->tstrs, tdims, CFL_SIZE);

	data->tensor = tensor;

	long odims[N];
	md_copy_dims(N, odims, data->odims);

	long idims[N];
	md_copy_dims(N, idims, data->idims);

	return linop_create(N, odims, N, idims,
			CAST_UP(PTR_PASS(data)), fmac_apply, fmac_adjoint, NULL,
			NULL, fmac_free_data);
}



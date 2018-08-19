/* Copyright 2014,2017. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 */
 
#include <complex.h>

#include "misc/mri.h"
#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/multind.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "linops/linop.h"

#include "sampling.h"


struct sampling_data_s {

	INTERFACE(linop_data_t);

	long dims[DIMS];
	long strs[DIMS];

	long pat_dims[DIMS];
	long pat_strs[DIMS];

	complex float* pattern;
#ifdef USE_CUDA
	const complex float* gpu_pattern;
#endif
};

static DEF_TYPEID(sampling_data_s);


#ifdef USE_CUDA
static const complex float* get_pat(const struct sampling_data_s* data, bool gpu)
{
	const complex float* pattern = data->pattern;

	if (gpu) {

		if (NULL == data->gpu_pattern)
			((struct sampling_data_s*)data)->gpu_pattern = md_gpu_move(DIMS, data->pat_dims, data->pattern, CFL_SIZE);

		pattern = data->gpu_pattern;
	}

	return pattern;
}
#endif

static void sampling_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(sampling_data_s, _data);

#ifdef USE_CUDA
	const complex float* pattern = get_pat(data, cuda_ondevice(src));
#else
	const complex float* pattern = data->pattern;
#endif

	md_zmul2(DIMS, data->dims, data->strs, dst, data->strs, src, data->pat_strs, pattern);
}

static void sampling_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(sampling_data_s, _data);

#ifdef USE_CUDA
	if (NULL != data->gpu_pattern) {
		md_free((void*)data->gpu_pattern);
	}
#endif

	xfree(data);
}

struct linop_s* linop_sampling_create(const long dims[DIMS], const long pat_dims[DIMS], const complex float* pattern)
{
	PTR_ALLOC(struct sampling_data_s, data);
	SET_TYPEID(sampling_data_s, data);

	md_copy_dims(DIMS, data->pat_dims, pat_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, data->dims, dims); // dimensions of kspace
	md_calc_strides(DIMS, data->strs, data->dims, CFL_SIZE);
	md_calc_strides(DIMS, data->pat_strs, data->pat_dims, CFL_SIZE);

	data->pattern = (complex float*)pattern;
#ifdef USE_CUDA
	data->gpu_pattern = NULL;
#endif

	const long* dims2 = data->dims;
	return linop_create(DIMS, dims2, DIMS, dims2, CAST_UP(PTR_PASS(data)), sampling_apply, sampling_apply, sampling_apply, NULL, sampling_free);
}



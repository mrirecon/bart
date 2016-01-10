/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */
 
#include <complex.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "linops/linop.h"

#include "sampling.h"


struct sampling_data_s {

	long dims[DIMS];
	long strs[DIMS];
	long pat_strs[DIMS];
	const complex float* pattern;
};


static void sampling_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct sampling_data_s* data = _data;
	md_zmul2(DIMS, data->dims, data->strs, dst, data->strs, src, data->pat_strs, data->pattern);
}


static void sampling_free(const void* _data)
{
	const struct sampling_data_s* data = _data;
	free((void*)data);
}

struct linop_s* sampling_create(const long dims[DIMS], const long pat_dims[DIMS], const complex float* pattern)
{
	PTR_ALLOC(struct sampling_data_s, data);

	md_select_dims(DIMS, ~MAPS_FLAG, data->dims, dims); // dimensions of kspace
	md_calc_strides(DIMS, data->strs, data->dims, CFL_SIZE);
	md_calc_strides(DIMS, data->pat_strs, pat_dims, CFL_SIZE);

	data->pattern = pattern;

	return linop_create(DIMS, data->dims, DIMS, data->dims, data, sampling_apply, sampling_apply, sampling_apply, NULL, sampling_free);
}



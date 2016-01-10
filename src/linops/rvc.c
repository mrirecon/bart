/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>

#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "linops/linop.h"

#include "rvc.h"


struct rvc_s {

	unsigned int N;
	const long* dims;
};

static void rvc_apply(const void* _data, complex float* dst, const complex float* src)
{
	const struct rvc_s* data = _data;
	md_zreal(data->N, data->dims, dst, src);
}

static void rvc_free(const void* _data)
{
	const struct rvc_s* data = _data;
	free((void*)data->dims);
	free((void*)data);
}


struct linop_s* rvc_create(unsigned int N, const long dims[N])
{
	PTR_ALLOC(struct rvc_s, data);

	PTR_ALLOC(long[N], dims2);
	md_copy_dims(N, *dims2, dims);

	data->N = N;
	data->dims = *dims2;

	return linop_create(N, dims, N, dims, (void*)data, rvc_apply, rvc_apply, rvc_apply, NULL, rvc_free);
}



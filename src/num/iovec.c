/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "iovec.h"


#if 1
void debug_print_iovec(int level, const struct iovec_s* vec)
{
	debug_printf(level, "iovec:\n");
	debug_printf(level, " N = %d\n", vec->N);
	debug_printf(level, " dims = \t");
	debug_print_dims(level, vec->N, vec->dims);
	debug_printf(level, " strs = \t");
	debug_print_dims(level, vec->N, vec->strs);
}
#endif


const struct iovec_s* iovec_create2(unsigned int N, const long dims[N], const long strs[N], size_t size)
{
	PTR_ALLOC(struct iovec_s, n);
	n->N = N;

	PTR_ALLOC(long[N], ndims);
	memcpy(*ndims, dims, N * sizeof(long));
	n->dims = *PTR_PASS(ndims);

	PTR_ALLOC(long[N], nstrs);
	memcpy(*nstrs, strs, N * sizeof(long));
	n->strs = *PTR_PASS(nstrs);

	n->size = size;

	return PTR_PASS(n);
}

const struct iovec_s* iovec_create(unsigned int N, const long dims[N], size_t size)
{
	long strs[N];
	md_calc_strides(N, strs, dims, size);
	return iovec_create2(N, dims, strs, size);
}


void iovec_free(const struct iovec_s* x)
{
	free((void*)x->dims);
	free((void*)x->strs);
	free((void*)x);
}

bool iovec_check(const struct iovec_s* iov, unsigned int N, const long dims[N], const long strs[N])
{
	bool ok = true;
	
	debug_print_dims(DP_DEBUG4, N, dims);
	debug_print_dims(DP_DEBUG4, iov->N, iov->dims);

	if (N != iov->N)
		return false;

	for (unsigned int i = 0; i < N; i++) {

		ok &= (dims[i] == iov->dims[i]);
		ok &= (strs[i] == iov->strs[i]);
	}

	return ok;
}




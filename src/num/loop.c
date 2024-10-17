/* Copyright 2014. The Regents of the University of California.
 * Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2020 Martin Uecker
 *
 * various functions built around md_loop
 * No GPU support at the moment!
 */

#include <complex.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/nested.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif
#include "num/multind.h"
#include "num/vptr.h"


#include "loop.h"


// typedef complex float (*sample_fun_t)(const long pos[]);

static void md_zsample2(int N, const long dims[N], unsigned long flags, complex float* out, zsample_fun_t fun)
{
	bool buf = is_vptr(out);
#ifdef USE_CUDA
	buf = buf || cuda_ondevice(out);
#endif

	if (buf) {

		complex float *out2 = md_alloc(N, dims, sizeof *out2);

		md_zsample2(N, dims, flags, out2, fun);
		md_copy(N, dims, out, out2, sizeof *out2);

		md_free(out2);
		return;
	}

	long strs[N];
	md_calc_strides(N, strs, dims, 1);	// we use size = 1 here

	long* strsp = strs;	// because of clang

	NESTED(void, sample_kernel, (const long pos[]))
	{
		out[md_calc_offset(N, strsp, pos)] = fun(pos);
	};

	md_parallel_loop(N, dims, flags, sample_kernel);
}

void md_zsample(int N, const long dims[N], complex float* out, zsample_fun_t fun)
{
	md_zsample2(N, dims, 0U, out, fun);
}

void md_parallel_zsample(int N, const long dims[N], complex float* out, zsample_fun_t fun)
{
	md_zsample2(N, dims, ~0U, out, fun);
}

static void md_zzsample2(int N, const long dims[N], unsigned long flags, complex double* out, zzsample_fun_t fun)
{
#ifdef USE_CUDA
	if (cuda_ondevice(out)) {

		complex double *out2 = md_alloc(N, dims, sizeof *out2);

		md_zzsample2(N, dims, flags, out2, fun);
		md_copy(N, dims, out, out2, sizeof *out2);

		md_free(out2);
		return;
	}
#endif

	long strs[N];
	md_calc_strides(N, strs, dims, 1);	// we use size = 1 here

	long* strsp = strs;	// because of clang

	NESTED(void, sample_kernel, (const long pos[]))
	{
		out[md_calc_offset(N, strsp, pos)] = fun(pos);
	};

	md_parallel_loop(N, dims, flags, sample_kernel);
}

void md_zzsample(int N, const long dims[N], complex double* out, zzsample_fun_t fun)
{
	md_zzsample2(N, dims, 0U, out, fun);
}

void md_parallel_zzsample(int N, const long dims[N], complex double* out, zzsample_fun_t fun)
{
	md_zzsample2(N, dims, ~0U, out, fun);
}

static void md_sample2(int N, const long dims[N], unsigned long flags, float* out, sample_fun_t fun)
{
	long strs[N];
	md_calc_strides(N, strs, dims, 1);	// we use size = 1 here

	long* strsp = strs;	// because of clang

	NESTED(void, sample_kernel, (const long pos[]))
	{
		out[md_calc_offset(N, strsp, pos)] = fun(pos);
	};

	md_parallel_loop(N, dims, flags, sample_kernel);
}

void md_sample(int N, const long dims[N], float* out, sample_fun_t fun)
{
	md_sample2(N, dims, 0U, out, fun);
}

void md_parallel_sample(int N, const long dims[N], float* out, sample_fun_t fun)
{
	md_sample2(N, dims, ~0U, out, fun);
}


void md_zmap(int N, const long dims[N], complex float* out, const complex float* in, map_fun_t fun)
{
	long strs[N];
	md_calc_strides(N, strs, dims, 1); // we use size = 1 here

	long* strsp = strs; // because of clang

	NESTED(complex float, map_kernel, (const long pos[]))
	{
		return fun(in[md_calc_offset(N, strsp, pos)]);
	};

	md_zsample(N, dims, out, map_kernel);
}


void md_zgradient(int N, const long dims[N], complex float* out, const complex float grad[N])
{
	long ndims[N];
	complex float ngrad[N];
	int nN = 0;

	for (int i = 0; i < N; i++) {

		if (1 != dims[i]) {

			ndims[nN] = dims[i];
			ngrad[nN] = grad[i];
			nN++;
		}
	}

	// clang
	const complex float* grad2 = ngrad;

	NESTED(complex float, gradient_kernel, (const long pos[]))
	{
		complex float val = 0.;

		for (int i = 0; i < nN; i++)
			val += pos[i] * grad2[i];

		return val;
	};

	md_parallel_zsample(nN, ndims, out, gradient_kernel);
}



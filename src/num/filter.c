/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"

#include "misc/misc.h"

#include "filter.h"


static int cmp_float(const void* a, const void* b)
{
	return (*(float*)a - *(float*)b > 0.) ? 1. : -1.;
}

static int cmp_complex_float(const void* a, const void* b) // gives sign for 0. (not 0)
{
	return (cabsf(*(complex float*)a) - cabsf(*(complex float*)b) > 0.) ? 1. : -1.;
}

static void sort_floats(int N, float ar[N])
{
	qsort((void*)ar, N, sizeof(float), cmp_float);
}

static void sort_complex_floats(int N, complex float ar[N])
{
	qsort((void*)ar, N, sizeof(complex float), cmp_complex_float);
}

float median_float(int N, float ar[N])
{
	float tmp[N];
	memcpy(tmp, ar, N * sizeof(float));
	sort_floats(N, tmp);
	return (1 == N % 2) ? tmp[(N - 1) / 2] : ((tmp[(N - 1) / 2 + 0] + tmp[(N - 1) / 2 + 1]) / 2.);
}

complex float median_complex_float(int N, complex float ar[N])
{
	complex float tmp[N];
	memcpy(tmp, ar, N * sizeof(complex float));
	sort_complex_floats(N, tmp);
	return (1 == N % 2) ? tmp[(N - 1) / 2] : ((tmp[(N - 1) / 2 + 0] + tmp[(N - 1) / 2 + 1]) / 2.);
}


struct median_s {

	long length;
	long stride;
};

static void nary_medianz(void* _data, void* ptr[])
{
	struct median_s* data = (struct median_s*)_data;

	complex float tmp[data->length];

	for (long i = 0; i < data->length; i++)
		tmp[i] = *((complex float*)(ptr[1] + i * data->stride));

	*(complex float*)ptr[0] = median_complex_float(data->length, tmp);
}

void md_medianz2(int D, int M, long dim[D], long ostr[D], complex float* optr, long istr[D], complex float* iptr)
{
	assert(M < D);
	const long* nstr[2] = { ostr, istr };
	void* nptr[2] = { optr, iptr };

	struct median_s data = { dim[M], istr[M] };

	long dim2[D];
	for (int i = 0; i < D; i++)
		dim2[i] = dim[i];

	dim2[M] = 1;

	md_nary(2, D, dim2, nstr, nptr, (void*)&data, &nary_medianz);
}

void md_medianz(int D, int M, long dim[D], complex float* optr, complex float* iptr)
{
	assert(M < D);

	long dim2[D];
	for (int i = 0; i < D; i++)
		dim2[i] = dim[i];

	dim2[M] = 1;

	long istr[D];
	long ostr[D];

	md_calc_strides(D, istr, dim, 8);
	md_calc_strides(D, ostr, dim2, 8);

	md_medianz2(D, M, dim, ostr, optr, istr, iptr);
}




void centered_gradient(unsigned int N, const long dims[N], const complex float grad[N], complex float* out)
{
	md_zgradient(N, dims, out, grad);

	long dims0[N];
	md_singleton_dims(N, dims0);

	long strs0[N];
	md_calc_strides(N, strs0, dims0, CFL_SIZE);

	complex float cn = 0.;

	for (unsigned int n = 0; n < N; n++)
		 cn -= grad[n] * (float)dims[n] / 2.;

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	md_zadd2(N, dims, strs, out, strs, out, strs0, &cn);
}

void linear_phase(unsigned int N, const long dims[N], const float pos[N], complex float* out)
{
	complex float grad[N];

	for (unsigned int n = 0; n < N; n++)
		grad[n] = 2.i * M_PI * (float)(pos[n]) / ((float)dims[n]);

	centered_gradient(N, dims, grad, out);
	md_zmap(N, dims, out, out, cexpf);
}


void klaplace_scaled(unsigned int N, const long dims[N], unsigned int flags, const float sc[N], complex float* out)
{
	unsigned int flags2 = flags;

	complex float* tmp = md_alloc(N, dims, CFL_SIZE);

	md_clear(N, dims, out, CFL_SIZE);

	for (unsigned int i = 0; i < bitcount(flags); i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		complex float grad[N];
		for (unsigned int j = 0; j < N; j++)
			grad[j] = 0.;

		grad[lsb] = sc[lsb];
		centered_gradient(N, dims, grad, tmp);
		md_zspow(N, dims, tmp, tmp, 2.);
		md_zadd(N, dims, out, out, tmp);
	}

	md_free(tmp);
}


void klaplace(unsigned int N, const long dims[N], unsigned int flags, complex float* out)
{
	float sc[N];
	for (unsigned int j = 0; j < N; j++)
		sc[j] = 1. / (float)dims[j];

	klaplace_scaled(N, dims, flags, sc, out);
}

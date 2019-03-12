/* Copyright 2019. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2019 Siddharth Iyer <ssi@mit.edu>
 */

#include <complex.h>

#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "linops/linop.h"

#include "decompose_complex.h"

#ifdef _OPENMP
#include <omp.h>
#endif

struct decompose_complex_s {
	INTERFACE(linop_data_t);

	unsigned int N;
	unsigned int D;
	unsigned int K;

	const long* idims;
	const long* odims;

  complex float* buffer;
};

static DEF_TYPEID(decompose_complex_s);

static void decompose_complex_fwd(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(decompose_complex_s, _data);

	#pragma omp parallel for
	for (long k = 0; k < data->K; k ++)
		dst[k] = creal(src[k]) + 1.0i * cimag(src[k + data->K]);
}

static void decompose_complex_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(decompose_complex_s, _data);
	md_zreal(data->N, data->odims, dst, src);
	md_zimag(data->N, data->odims, dst + data->K, src);
}

static void decompose_complex_nrm(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const auto data = CAST_DOWN(decompose_complex_s, _data);
	md_copy(data->N, data->idims, dst, src, sizeof(complex float));  // Identity.
}

static void decompose_complex_free(const linop_data_t* _data)
{
	const auto data = CAST_DOWN(decompose_complex_s, _data);

	xfree(data->idims);
	xfree(data->odims);
	xfree(data);
}

struct linop_s* linop_decompose_complex_create(unsigned int N, unsigned int D, const long dims[N])
{
	assert(D < N);
	for (long k = D; k < N; k++)
		assert(1 == dims[k]);

	long K = 1;
	for (long k = 0; k < D; k++)
		K = K * dims[k];

	PTR_ALLOC(struct decompose_complex_s, data);
	SET_TYPEID(decompose_complex_s, data);

	long idims[N];
	md_copy_dims(N, idims, dims);
	idims[D] = 2;
	long odims[N];
	md_copy_dims(N, odims, dims);

	PTR_ALLOC(long[N], idims_alloc);
	PTR_ALLOC(long[N], odims_alloc);

	md_copy_dims(N, *idims_alloc, idims);
	md_copy_dims(N, *odims_alloc, odims);

	data->N = N;
	data->D = D;
	data->K = K;
	data->idims = *PTR_PASS(idims_alloc);
	data->odims = *PTR_PASS(odims_alloc);

	return linop_create(N, odims, N, idims, CAST_UP(PTR_PASS(data)), decompose_complex_fwd, decompose_complex_adj, decompose_complex_nrm, NULL, decompose_complex_free);
}

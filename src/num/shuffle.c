/* Copyright 2013-2015 The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include "num/multind.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/debug.h"
#include "misc/misc.h"

#include "shuffle.h"


#if 0
void md_shuffle2(unsigned int N, const long dims[N], const long factors[N],
		const long ostrs[N], void* out, const long istrs[N], const void* in, size_t size)
{
	long dims2[2 * N];
	long ostrs2[2 * N];
	long istrs2[2 * N];

	for (unsigned int i = 0; i < N; i++) {

		assert(0 == dims[i] % factors[i]);

		long f2 = dims[i] / factors[i];

		dims2[0 * N + i] = f2;
		dims2[1 * N + i] = factors[i];

		ostrs2[1 * N + i] = ostrs[i];
		ostrs2[0 * N + i] = ostrs[i] * f2;

		istrs2[0 * N + i] = istrs[i] * factors[i];
		istrs2[1 * N + i] = istrs[i];
	}

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_shuffle(unsigned int N, const long dims[N], const long factors[N],
		void* out, const void* in, size_t size)
{
	long strs[N];
	md_calc_strides(N, strs, dims, size);

	md_shuffle2(N, dims, factors, strs, out, strs, in, size);
}
#endif


static void decompose_dims(unsigned int N, long dims2[2 * N], long ostrs2[2 * N], long istrs2[2 * N],
		const long factors[N], const long odims[N + 1], const long ostrs[N + 1], const long idims[N], const long istrs[N])
{
	long prod = 1;

	for (unsigned int i = 0; i < N; i++) {

		long f2 = idims[i] / factors[i];

		assert(0 == idims[i] % factors[i]);
		assert(odims[i] == idims[i] / factors[i]);

		dims2[1 * N + i] = factors[i];
		dims2[0 * N + i] = f2;

		istrs2[0 * N + i] = istrs[i] * factors[i];
		istrs2[1 * N + i] = istrs[i];

		ostrs2[0 * N + i] = ostrs[i];
		ostrs2[1 * N + i] = ostrs[N] * prod;

		prod *= factors[i];
	}

	assert(odims[N] == prod);
}

void md_decompose2(unsigned int N, const long factors[N],
		const long odims[N + 1], const long ostrs[N + 1], void* out,
		const long idims[N], const long istrs[N], const void* in, size_t size)
{
	long dims2[2 * N];
	long ostrs2[2 * N];
	long istrs2[2 * N];

	decompose_dims(N, dims2, ostrs2, istrs2, factors, odims, ostrs, idims, istrs);

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_decompose(unsigned int N, const long factors[N], const long odims[N + 1],
		void* out, const long idims[N], const void* in, size_t size)
{
#ifdef USE_CUDA

	int n_factors = bitcount(md_nontriv_dims(N, factors));

	if (cuda_ondevice(out) && (1 < n_factors)) {

		//do decomposition for each dimension independently
		//=> much less calls to cuda copy strided
		//FIXME: we should have generic strided copy kernel on GPU! 

		void* tmp = md_alloc_sameplace(N, idims, size, in);

		void* tmp_dst = (1 == n_factors % 2) ? out : tmp;
		void* tmp_src = (void*)in;

		long factors2[2 * N];
		long idims2[2 * N];
		long odims2[2 * N + 1];

		md_copy_dims(N, idims2, idims);
		md_singleton_dims(N, idims2 + N);

		md_copy_dims(N, odims2, idims);
		md_singleton_dims(N, odims2 + N);

		md_singleton_dims(2 * N, factors2);

		for (unsigned int i = 0; i < N; i++) {

			if (1 < factors[i]) {

				odims2[i] /= factors[i];
				odims2[2 * N] = factors[i];
				factors2[i] = factors[i];

				md_decompose(2 * N, factors2, odims2, tmp_dst, idims2, tmp_src, size);

				if (tmp_src == in)
					tmp_src = (0 == n_factors % 2) ? out : tmp;

				SWAP(tmp_dst, tmp_src);

				factors2[i] = 1;
				idims2[i] /= factors[i];

				idims2[N + i] = factors[i];
				odims2[N + i] = factors[i];
			}
		}

		md_free(tmp);

		return;
	}
#endif

	long ostrs[N + 1];
	md_calc_strides(N + 1, ostrs, odims, size);

	long istrs[N];
	md_calc_strides(N, istrs, idims, size);

	md_decompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
}

void md_recompose2(unsigned int N, const long factors[N],
		const long odims[N], const long ostrs[N], void* out,
		const long idims[N + 1], const long istrs[N + 1], const void* in, size_t size)
{
	long dims2[2 * N];
	long ostrs2[2 * N];
	long istrs2[2 * N];

	decompose_dims(N, dims2, istrs2, ostrs2, factors, idims, istrs, odims, ostrs);

	md_copy2(2 * N, dims2, ostrs2, out, istrs2, in, size);
}

void md_recompose(unsigned int N, const long factors[N], const long odims[N],
		void* out, const long idims[N + 1], const void* in, size_t size)
{
	#ifdef USE_CUDA

	int n_factors = bitcount(md_nontriv_dims(N, factors));

	if (cuda_ondevice(out) && (1 < n_factors)) {

		//do recomposition for each dimension independently
		//=> much less calls to cuda copy strided
		//FIXME: we should have generic strided copy kernel on GPU!

		void* tmp = md_alloc_sameplace(N + 1, idims, size, in);

		void* tmp_dst = (1 == n_factors % 2) ? out : tmp;
		void* tmp_src = (void*)in;

		long factors2[2 * N];
		long idims2[2 * N + 1];
		long odims2[2 * N];

		md_copy_dims(N, idims2, idims);
		md_copy_dims(N, idims2 + N, factors);

		md_copy_dims(N, odims2, idims);
		md_copy_dims(N, odims2 + N, factors);

		md_singleton_dims(2 * N, factors2);

		for (int i = N - 1; i >= 0; i--) {

			if (1 < factors[i]) {

				idims2[2 * N] = factors[i];
				odims2[i] *= factors[i];
				
				factors2[i] = factors[i];
				idims2[N + i] = 1;
				odims2[N + i] = 1;

				md_recompose(2 * N, factors2, odims2, tmp_dst, idims2, tmp_src, size);

				if (tmp_src == in)
					tmp_src = (0 == n_factors % 2) ? out : tmp;

				SWAP(tmp_dst, tmp_src);

				factors2[i] = 1;
				idims2[i] *= factors[i];
			}
		}

		md_free(tmp);

		return;
	}
#endif


	long ostrs[N];
	md_calc_strides(N, ostrs, odims, size);

	long istrs[N + 1];
	md_calc_strides(N + 1, istrs, idims, size);

	md_recompose2(N, factors, odims, ostrs, out, idims, istrs, in, size);
}


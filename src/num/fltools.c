/* Copyright 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/nested.h"
#include "misc/misc.h"

#include "fltools.h"

static void msort(void* tmp, size_t N1, size_t N2, size_t size, __compar_fn_t compare)
{
	if ((0 == N1) || (0 == N2))
		return;
		
	void* tmp1 = xmalloc(N1 * size);
	memcpy(tmp1, tmp, N1 * size);

	void* tmp2 = tmp + N1 * size;

	size_t i1 = 0;
	size_t i2 = 0;

	size_t i = 0;
	for (; i < N1 + N2; i++) {

		if ((i1 == N1) || (i2 == N2))
			break;

		if (0 > compare(tmp1 + i1 * size, tmp2 + i2 * size))
			memcpy(tmp + i * size, tmp1 + (i1++) * size, size);
		else
			memcpy(tmp + i * size, tmp2 + (i2++) * size, size);
	}

	if (i1 < N1)
		memcpy(tmp + i * size, tmp1 + i1 * size, (N1 - i1) * size);
	
	xfree(tmp1);
}


static void psort(void* tmp, size_t N, size_t size, __compar_fn_t compare)
{
	int threads = 1;
#ifdef _OPENMP
	threads = omp_get_max_threads();
#endif
	int split = 1;
	while (split < threads)
		split *= 2;

	size_t sizes[split];
	size_t offset[split];

	size_t Np = N;

	for (int i = split; i > 0; i--) {

		if (i > threads) {

			sizes[i - 1] = 0;
		} else {

			sizes[i - 1] = Np / (size_t)i;
			Np -= sizes[i - 1];
		}
	}

	offset[0] = 0;
	for(int i = 1; i < split; i++)
		offset[i] = offset[i - 1] + sizes[i - 1];

#pragma omp parallel for
	for (int i = 0; i < split; i++)
		qsort(tmp + offset[i] * size, (size_t)sizes[i], size, compare);


	while (1 < split) {

#pragma omp parallel for
		for (int i = 0; i < split / 2; i++)
			msort(tmp + offset[2 * i] * size, sizes[2 * i], sizes[2 * i + 1], size, compare);

		for (int i = 0; i < split / 2; i++) {

			sizes[i] = sizes[2 * i] + sizes[2 * i + 1];
			offset[i] = offset[2 * i];
		}

		split /= 2;	
	}
}

static int compare(const void* _a, const void* _b)
{
	const complex float* a = _a;
	const complex float* b = _b;
	return copysignf(1., (cabsf(*a) - cabsf(*b)));
}

void zsort(int N, complex float tmp[N])
{
	psort(tmp, (size_t)N, sizeof(complex float), compare);
}

complex float zselect(int N, int k, const complex float x[N])
{
	complex float (*p)[N] = xmalloc(sizeof(*p));

	memcpy(*p, x, sizeof *p);

	zsort(N, *p);

	complex float val = (*p)[k];

	xfree(p);

	return val;
}


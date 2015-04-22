/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 *
 * 
 * Optimization framework for operations on multi-dimensional arrays.
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <stdio.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#ifdef BERKELEY_SVN
#include "num/simplex.h"
#endif

#include "optimize.h"

/*
 * Helper functions:
 *
 * 1. detect aliasing
 * 2. detect if dimensions can be merged
 * 3. compute memory footprint
 *
 */


#if 0
static bool regular(long dim, long str)
{
	return (dim > 0) && (str > 0);	
}

static bool singular(long dim, long str)
{
	assert(dim > 0);

	return (1 == dim) || (0 == str);
}

static bool enclosed(const long dims[2], const long strs[2])
{
	assert(regular(dims[0], strs[0]));
	assert(regular(dims[1], strs[1]));

	return (strs[1] >= dims[0] * strs[0]);
}




// assumes no overlap
static long memory_footprint(int N, const long dims[N], const long strs[N])
{
	unsigned int flags = 0;

	for (int i = 0; i < N; i++)
		flags |= (0 == strs[i]);

	long dims2[N];
	md_select_dims(N, ~flags, dims2, dims);
	return md_calc_size(N, dims2);
}

#endif

/*
 * Generic optimizations strategy:
 *
 * 1. ordering of dimensions by stride
 * 2. merging of dimensions
 * 3. splitting and ordering (cache-oblivious algorithms)
 * 4. parallelization
 * 
 */

/* strategies:

        - cache-oblivous algorithms (e.g. transpose)
        - use of accelerators 
        - parallelization
        - vectorization
        - reordering of memory access
        - temporaries
        - loop merging
        - splitting
*/      

/*
 * Each parameter is either input or output. The pointers must valid
 * and all accesses using any position inside the range given by
 * dimensions and using corresponding strides must be inside of the 
 * adressed memory region. Pointers pointing inside the same region
 * can be passed multipe times.
 */



void merge_dims(unsigned int D, unsigned int N, long dims[N], long (*ostrs[D])[N])
{
	for (int i = N - 2; i >= 0; i--) {

		bool domerge = true;

		for (unsigned int j = 0; j < D; j++) // mergeable
			domerge &= (*ostrs[j])[i + 1] == dims[i] * (*ostrs[j])[i];

		if (domerge) {

			for (unsigned int j = 0; j < D; j++)
				(*ostrs[j])[i + 1] = 0;

			dims[i + 0] *= dims[i + 1];
			dims[i + 1] = 1;
		}
	}
}


unsigned int remove_empty_dims(unsigned int D, unsigned int N, long dims[N], long (*ostrs[D])[N])
{
	unsigned int o = 0;

	for (unsigned int i = 0; i < N; i++) {

		if (1 != dims[i]) {

			dims[o] = dims[i];
			
			for (unsigned int j = 0; j < D; j++)
				(*ostrs[j])[o] = (*ostrs[j])[i];
			o++;
		}
	}

	return o;
}


static int cmp_strides(const void* _data, unsigned int a, unsigned int b)
{
	const long* strs = _data;
	long d = strs[a] - strs[b];

	if (d > 0)
		return 1;
	if (d < 0)
		return -1;

	return 0;
}

static void compute_permutation(unsigned int N, unsigned int ord[N], const long strs[N])
{
	for (unsigned int i = 0; i < N; i++)
		ord[i] = i;

	quicksort(N, ord, (const void*)strs, cmp_strides);
}

static void reorder_long(unsigned int N, unsigned int ord[N], long x[N])
{
	long tmp[N];
	memcpy(tmp, x, N * sizeof(long));

	for (unsigned int i = 0; i < N; i++)
		x[i] = tmp[ord[i]];
}


#ifdef BERKELEY_SVN
/*
 * Jim Demmel's generic blocking theorem
 */
static void demmel_factors(unsigned int D, unsigned int N, float blocking[N], long (*strs[D])[N])
{
	float delta[D][N];

	for (unsigned int d = 0; d < D; d++)
		for (unsigned int n = 0; n < N; n++)
			delta[d][n] = (0 != (*strs[d])[n]) ? 1. : 0.;


	// now maximize 1^T x subject to Delta x <= 1
	// M^{x_n} yields blocking factors where M is cache size (maybe needs to be devided by D?)

	float ones[MAX(N, D)];
	for (unsigned int n = 0; n < MAX(N, D); n++)
		ones[n] = 1.;

	simplex(D, N, blocking, ones, ones, (const float (*)[N])delta);
}
#endif


static long find_factor(long x, float blocking)
{
	//long m = (long)(1. + sqrt((double)x));
	long m = (long)(1. + pow((double)x, blocking));

	for (long i = m; i > 1; i--)
		if (0 == x % i)
			return (x / i);

	return 1;
}


static bool split_dims(unsigned int D, unsigned int N, long dims[N + 1], long (*ostrs[D])[N + 1], float blocking[N])
{
	if (0 == N)
		return false;

	long f;
	if ((dims[N - 1] > 1024) && (1 < (f = find_factor(dims[N - 1], blocking[N - 1])))) {
//	if ((dims[N - 1] > 64) && (1 < (f = find_factor(dims[N - 1], blocking[N - 1])))) {
#if 1
		//printf("Split with %ld=%ldx%ld\n", dims[N - 1], f, dims[N - 1] / f);

		dims[N] = dims[N - 1] / f;
		dims[N - 1] = f;

		for (unsigned int j = 0; j < D; j++)
			(*ostrs[j])[N] = (*ostrs[j])[N - 1] * f;

		blocking[N] = 1.;
#else
		dims[N] = 1;
		for (unsigned int j = 0; j < D; j++)
			(*ostrs[j])[N] = 0;
#endif
		return true;
	}

	// could not split, make room and try lower dimensions

	dims[N] = dims[N - 1];
	blocking[N] = blocking[N - 1];

	for (unsigned int j = 0; j < D; j++)
		(*ostrs[j])[N] = (*ostrs[j])[N - 1];

	if (split_dims(D, N - 1, dims, ostrs, blocking))
		return true;

	dims[N - 1] = dims[N];

	for (unsigned int j = 0; j < D; j++)
		(*ostrs[j])[N - 1] = (*ostrs[j])[N];

	blocking[N - 1] = blocking[N];

	return false;
}


unsigned int optimize_dims(unsigned int D, unsigned int N, long dims[N], long (*strs[D])[N])
{
	merge_dims(D, N, dims, strs);

	unsigned int ND = remove_empty_dims(D, N, dims, strs);

	if (0 == ND) { // atleast return a single dimension

		dims[0] = 1;
		
		for (unsigned int j = 0; j < D; j++)
			(*strs[j])[0] = 0;

		ND = 1;
	}

	debug_print_dims(DP_DEBUG4, ND, dims);

	float blocking[N];
#ifdef BERKELEY_SVN
	// actually those are not the blocking factors
	// as used below but relative to fast memory
	//demmel_factors(D, ND, blocking, strs);
	UNUSED(demmel_factors);
#endif
#if 0
	debug_printf(DP_DEBUG4, "DB: ");
	for (unsigned int i = 0; i < ND; i++)
		debug_printf(DP_DEBUG4, "%f\t", blocking[i]);
	debug_printf(DP_DEBUG4, "\n");
#endif
#if 1
	for (unsigned int i = 0; i < ND; i++)
		blocking[i] = 0.5;
	//	blocking[i] = 1.;
#endif

	// try to split dimensions according to blocking factors
	// use space up to N

	bool split = false;

	do {
		if (N == ND)
			break;

		split = split_dims(D, ND, dims, strs, blocking);

		if (split)
			ND++;

	} while(split);

//	printf("Split %c :", split ? 'y' : 'n');
//	print_dims(ND, dims);

	long max_strides[ND];

	for (unsigned int i = 0; i < ND; i++) {

		max_strides[i] = 0;

		for (unsigned int j = 0; j < D; j++)
			max_strides[i] = MAX(max_strides[i], (*strs[j])[i]);
	}

	unsigned int ord[ND];
	compute_permutation(ND, ord, max_strides);

//	for (unsigned int i = 0; i < ND; i++)
//		printf("%d: %ld %d\n", i, max_strides[i], ord[i]);
#if 1
	for (unsigned int j = 0; j < D; j++)
		reorder_long(ND, ord, *strs[j]);

	reorder_long(ND, ord, dims);
#endif

#if 0
	printf("opt dims\n");
	print_dims(ND, dims);
	if (D > 0)
		print_dims(ND, *strs[0]);
	if (D > 1)
		print_dims(ND, *strs[1]);
	if (D > 2)
		print_dims(ND, *strs[2]);
#endif

	return ND;
}



/**
 * compute minimal dimension of largest contiguous block(s)
 *
 */
unsigned int min_blockdim(unsigned int D, unsigned int N, const long dims[N], long (*strs[D])[N], size_t size[D])
{
	unsigned int mbd = N;

	for (unsigned int i = 0; i < D; i++)
		mbd = MIN(mbd, md_calc_blockdim(N, dims, *strs[i], size[i]));

	return mbd;
}



static void compute_enclosures(unsigned int N, bool matrix[N][N], const long dims[N], const long strides[N])
{
	long ext[N];

	for (unsigned int i = 0; i < N; i++)
		ext[i] = dims[i] * labs(strides[i]);

	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++)
			matrix[i][j] = (ext[i] <= labs(strides[j]));
}


/**
 * compute set of parallelizable dimensions
 *
 */
static unsigned int parallelizable(unsigned int D, unsigned int io, unsigned int N, const long dims[N], long (*strs[D])[N], size_t size[D])
{
	// we assume no input / output overlap
	// (i.e. inputs which are also outputs have to be marked as output)

	// a dimension is parallelizable if all output operations
	// for that dimension are independent

	// for all output operations:
	// check - all other dimensions have strides greater or equal
	// the extend of this dimension or have an extend smaller or
	// equal the stride of this dimension

	// no overlap: [222]
	//                   [111111111111]
	//                                [333333333]
	//    overlap: [222]
	//		     [1111111111111111]
	//                                [333333333]

	unsigned int flags = (1 << N) - 1;

	for (unsigned int d = 0; d < D; d++) {

		if (MD_IS_SET(io, d)) {

			bool m[N][N];
			compute_enclosures(N, m, dims, *strs[d]);

	//		print_dims(N, dims);
	//		print_dims(N, *strs[d]);

			for (unsigned int i = 0; i < N; i++) {

				unsigned int a = 0;

				for (unsigned int j = 0; j < N; j++)
					if (m[i][j] || m[j][i])
						a++;

	//			printf("%d %d %d\n", d, i, a);

				if ((a != N - 1) || ((size_t)labs((*strs[d])[i]) < size[d]))
					flags = MD_CLEAR(flags, i);
			}
		}
	}

	return flags;
}


#define CHUNK (32 * 1024)
#define CORES (64)


/**
 * compute set of dimensions to parallelize
 *
 */
unsigned int dims_parallel(unsigned int D, unsigned int io, unsigned int N, const long dims[N], long (*strs[D])[N], size_t size[D])
{
	unsigned int flags = parallelizable(D, io, N, dims, strs, size);

	unsigned int i = D;
	unsigned int count = 1;

	long reps = md_calc_size(N, dims);

	unsigned int oflags = 0;

	while ((count < CORES) && (i-- > 0)) {

		if (MD_IS_SET(flags, i)) {

			reps /= dims[i];

			if (reps < CHUNK)
				break;

			oflags = MD_SET(oflags, i);

			//break; // only 1
		}
	}

	return oflags;
}






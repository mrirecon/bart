/* Copyright 2014. The Regents of the University of California.
 * Copyright 2016-2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 *
 * Optimization framework for operations on multi-dimensional arrays.
 */

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/nested.h"

#include "num/multind.h"
#include "num/vecops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls_copy.h"
#endif
#include "num/simplex.h"
#include "num/mpi_ops.h"
#include "num/vptr.h"

#include "optimize.h"

#ifdef __MINGW32__
#define ffs __builtin_ffs
#endif

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
	unsigned long flags = 0;

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
 * addressed memory region. Pointers pointing inside the same region
 * can be passed multiple times.
 */



void merge_dims(int D, int N, long dims[N], long (*ostrs[D])[N])
{
	for (int i = N - 2; i >= 0; i--) {

		bool domerge = true;

		for (int j = 0; j < D; j++) // mergeable
			domerge = domerge && ((*ostrs[j])[i + 1] == dims[i] * (*ostrs[j])[i]);

		if (domerge) {

			for (int j = 0; j < D; j++)
				(*ostrs[j])[i + 1] = 0;

			dims[i + 0] *= dims[i + 1];
			dims[i + 1] = 1;
		}

		if (1 == dims[i + 0]) { //everything can be merged with an empty dimension

			dims[i + 0] = dims[i + 1];
			dims[i + 1] = 1;

			for (int j = 0; j < D; j++) {

				(*ostrs[j])[i + 0] = (*ostrs[j])[i + 1];
				(*ostrs[j])[i + 1] = 0;
			}
		}
	}
}


int remove_empty_dims(int D, int N, long dims[N], long (*ostrs[D])[N])
{
	int o = 0;

	for (int i = 0; i < N; i++) {

		if (1 != dims[i]) {

			dims[o] = dims[i];

			for (int j = 0; j < D; j++)
				(*ostrs[j])[o] = (*ostrs[j])[i];
			o++;
		}
	}

	for (int i = o; i < N; i++) {

		for (int j = 0; j < D; j++)
			(*ostrs[j])[i] = 0;

		dims[i] = 1;
	}

	return o;
}




static void compute_permutation(int N, int ord[N], const long strs[N])
{
	__block const long* strsp = strs; // clang workaround

	for (int i = 0; i < N; i++)
		ord[i] = i;

	NESTED(int, cmp_strides, (int a, int b))
	{
		long da = strsp[a];
		long db = strsp[b];

		return (da > db) - (da < db);
	};

	quicksort(N, ord, cmp_strides);
}

static void reorder_long(int N, int ord[N], long x[N])
{
	long tmp[N];
	memcpy(tmp, x, (size_t)(N * (long)sizeof(long)));

	for (int i = 0; i < N; i++)
		x[i] = tmp[ord[i]];
}


/*
 * Jim Demmel's generic blocking theorem
 */
static void demmel_factors(int D, int N, float blocking[N], long (*strs[D])[N])
{
	float delta[D][N];

	for (int d = 0; d < D; d++)
		for (int n = 0; n < N; n++)
			delta[d][n] = (0 != (*strs[d])[n]) ? 1. : 0.;


	// now maximize 1^T x subject to Delta x <= 1
	// M^{x_n} yields blocking factors where M is cache size (maybe needs to be divided by D?)

	float ones[MAX(N, D)];
	for (int n = 0; n < MAX(N, D); n++)
		ones[n] = 1.;

	simplex(D, N, blocking, ones, ones, (const float (*)[N])delta);
}


static long find_factor(long x, float blocking)
{
	//long m = (long)(1. + sqrt((double)x));
	long m = (long)(1. + pow((double)x, blocking));

	for (long i = m; i > 1; i--)
		if (0 == x % i)
			return (x / i);

	return 1;
}


static bool split_dims(int D, int N, long dims[N + 1], long (*ostrs[D])[N + 1], float blocking[N + 1])
{
	if (0 == N)
		return false;

	long f;
	if ((dims[N - 1] > 1024) && (1 < (f = find_factor(dims[N - 1], blocking[N - 1])))) {
#if 1
		dims[N - 1] = dims[N - 1] / f;
		dims[N] = f;

		for (int j = 0; j < D; j++)
			(*ostrs[j])[N] = (*ostrs[j])[N - 1] * dims[N - 1];

		blocking[N - 1] = blocking[N - 1];
		blocking[N] = blocking[N - 1];
#else
		dims[N] = 1;
		for (int j = 0; j < D; j++)
			(*ostrs[j])[N] = 0;
#endif
		return true;
	}

	// could not split, make room and try lower dimensions

	dims[N] = dims[N - 1];
	blocking[N] = blocking[N - 1];

	for (int j = 0; j < D; j++)
		(*ostrs[j])[N] = (*ostrs[j])[N - 1];

	if (split_dims(D, N - 1, dims, ostrs, blocking))
		return true;

	dims[N - 1] = dims[N];

	for (int j = 0; j < D; j++)
		(*ostrs[j])[N - 1] = (*ostrs[j])[N];

	blocking[N - 1] = blocking[N];

	return false;
}



int simplify_dims(int D, int N, long dims[N], long (*strs[D])[N])
{
	merge_dims(D, N, dims, strs);

	int ND = remove_empty_dims(D, N, dims, strs);

	if (0 == ND) { // at least return a single dimension

		dims[0] = 1;

		for (int j = 0; j < D; j++)
			(*strs[j])[0] = 0;

		ND = 1;
	}

	return ND;
}


int optimize_dims(int D, int N, long dims[N], long (*strs[D])[N])
{
	int ND = simplify_dims(D, N, dims, strs);

	debug_print_dims(DP_DEBUG4, ND, dims);

	float blocking[N];
	// actually those are not the blocking factors
	// as used below but relative to fast memory
	//demmel_factors(D, ND, blocking, strs);
	(void)demmel_factors;
#if 0
	debug_printf(DP_DEBUG4, "DB: ");
	for (int i = 0; i < ND; i++)
		debug_printf(DP_DEBUG4, "%f\t", blocking[i]);
	debug_printf(DP_DEBUG4, "\n");
#endif
#if 1
	for (int i = 0; i < ND; i++)
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

	} while (split);

//	printf("Split %c :", split ? 'y' : 'n');
//	print_dims(ND, dims);

	long max_strides[ND];

	for (int i = 0; i < ND; i++) {

		max_strides[i] = 0;

		for (int j = 0; j < D; j++)
			max_strides[i] = MAX(max_strides[i], (*strs[j])[i]);
	}

	int ord[ND];
	compute_permutation(ND, ord, max_strides);

//	for (int i = 0; i < ND; i++)
//		printf("%d: %ld %d\n", i, max_strides[i], ord[i]);
#if 1
	for (int j = 0; j < D; j++)
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

int optimize_dims_gpu(int D, int N, long dims[N], long (*strs[D])[N])
{
	int ND = simplify_dims(D, N, dims, strs);

	debug_print_dims(DP_DEBUG4, ND, dims);

	long max_strides[ND];

	for (int i = 0; i < ND; i++) {

		max_strides[i] = 0;

		for (int j = 0; j < D; j++)
			max_strides[i] = MAX(max_strides[i], (*strs[j])[i]);
	}

	int ord[ND];
	compute_permutation(ND, ord, max_strides);

#if 1
	for (int j = 0; j < D; j++)
		reorder_long(ND, ord, *strs[j]);

	reorder_long(ND, ord, dims);
#endif

	ND = simplify_dims(D, N, dims, strs);
	
	return ND;
}




/**
 * compute minimal dimension of largest contiguous block(s)
 *
 */
int min_blockdim(int D, int N, const long dims[N], long (*strs[D])[N], size_t size[D])
{
	int mbd = N;

	for (int i = 0; i < D; i++)
		mbd = MIN(mbd, md_calc_blockdim(N, dims, *strs[i], size[i]));

	return mbd;
}



static void compute_enclosures(int N, bool matrix[N][N], const long dims[N], const long strides[N])
{
	long ext[N];

	for (int i = 0; i < N; i++)
		ext[i] = dims[i] * labs(strides[i]);

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			matrix[i][j] = (ext[i] <= labs(strides[j]));
}


/**
 * compute set of parallelizable dimensions
 *
 */
unsigned long parallelizable(int D, unsigned int io, int N, const long dims[N], const long (*strs[D])[N], size_t size[D])
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

	unsigned long flags = (1UL << N) - 1;

	for (int d = 0; d < D; d++) {

		if (MD_IS_SET(io, d)) {

			bool m[N][N];
			compute_enclosures(N, m, dims, *strs[d]);

	//		print_dims(N, dims);
	//		print_dims(N, *strs[d]);

			for (int i = 0; i < N; i++) {

				int a = 0;

				for (int j = 0; j < N; j++)
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


extern long num_chunk_size;
long num_chunk_size = 32 * 256;


/**
 * compute set of dimensions to parallelize
 *
 */
unsigned long dims_parallel(int D, unsigned long io, int N, const long dims[N], long (*strs[D])[N], size_t size[D])
{
	unsigned long flags = parallelizable(D, io, N, dims, (const long (**)[])strs, size);

	int i = N;

	long max_size = 0;
	for (int i = 0; i < D; i++)
		max_size = MAX(max_size, (long)size[i]);

	long reps = md_calc_size(N, dims) * max_size;

	unsigned long oflags = 0;

	while (i-- > 0) {

		if (MD_IS_SET(flags, i)) {

			reps /= dims[i];

			if (reps < num_chunk_size)
				break;

			oflags = MD_SET(oflags, i);
		}
	}

	return oflags;
}


#ifdef USE_CUDA
static bool use_gpu(int p, void* ptr[p])
{
	bool gpu = false;

	for (int i = 0; i < p; i++)
		gpu = gpu || cuda_ondevice(ptr[i]);

	for (int i = 0; i < p; i++)
		gpu = gpu && cuda_ondevice(ptr[i]);

#if 0
	// FIXME: fails for copy
	if (!gpu) {

		for (int i = 0; i < p; i++)
			assert(!cuda_ondevice(ptr[i]));
	}
#endif
	return gpu;
}

static bool one_on_gpu(int p, void* ptr[p])
{
	bool gpu = false;

	for (int i = 0; !gpu && (i < p); i++)
		gpu |= cuda_ondevice(ptr[i]);

	return gpu;
}
#endif

extern double md_flp_total_time;
double md_flp_total_time = 0.;

// automatic parallelization
extern bool num_auto_parallelize;
bool num_auto_parallelize = true;






/**
 * Optimized n-op.
 *
 * @param N number of arguments
 ' @param io bitmask indicating input/output
 * @param D number of dimensions
 * @param dim dimensions
 * @param nstr strides for arguments and dimensions
 * @param nptr argument pointers
 * @param sizes size of data for each argument, e.g. complex float
 * @param too n-op function
 * @param data_ptr pointer to additional data used by too
 */
void optimized_nop(int N, unsigned long io, int D, const long dim[D], const long (*nstr[N])[D?:1], void* const nptr[N], size_t sizes[N], md_nary_opt_fun_t too)
{
	assert(N > 0);

	if (0 == D) {

		long dim1[1] = { 1 };
		long tstrs[N][1];
		long (*nstr1[N])[1];

		for (int i = 0; i < N; i++) {

			tstrs[i][0] = 0;
			nstr1[i] = &tstrs[i];
		}

		optimized_nop(N, io, 1, dim1, (void*)nstr1, nptr, sizes, too);

		return;
	}

	vptr_assert_sameplace(N, (void**)nptr);

	bool mpi = false;

	for (int i = 0; i < N; ++i)
		mpi = mpi || is_mpi(nptr[i]);

	if (mpi) {

		unsigned long mpi_flags = 0UL;

		for (int i = 0; i < N; ++i)
			mpi_flags |= vptr_block_loop_flags(D, dim, (long*)nstr[i], nptr[i], sizes[i]);

		long ldims[D];
		long bdims[D];

		md_select_dims(D, ~mpi_flags, bdims, dim);
		md_select_dims(D, mpi_flags, ldims, dim);

		const long* bdimsp = bdims;
		size_t* sizesp = sizes;
		void* nstrp = nstr;

		NESTED(void, nary_mpi_optimize, (void* ptr[]))
		{
			optimized_nop(N, io, D, bdimsp, nstrp, ptr, sizesp, too);
		};

		md_nary(N, D, ldims, (void*)nstr, (void*)nptr, nary_mpi_optimize);

		return;
	}

	long tdims[D];
	md_copy_dims(D, tdims, dim);

	long tstrs[N][D];
	long (*nstr1[N])[D];
	void* nptr1[N];

	for (int i = 0; i < N; i++) {

		md_copy_strides(D, tstrs[i], *nstr[i]);

		nstr1[i] = &tstrs[i];
		nptr1[i] = vptr_resolve(nptr[i]);
	}

	bool gpu = false;

#ifdef USE_CUDA
	gpu = use_gpu(N, nptr1);
	int ND = (gpu ? optimize_dims_gpu : optimize_dims)(N, D, tdims, nstr1);
#else
	int ND = optimize_dims(N, D, tdims, nstr1);
#endif

	void* cnst_buf[N];
	for (int i = 0; i < N; i++)
		cnst_buf[i] = NULL;

#if 1
	int NB = ND;
	for (int i = 0; i < N; i++)
		NB = MIN(NB, md_calc_blockdim(ND, tdims, tstrs[i], sizes[i]));

	unsigned long cnst_flags = 0;
	bool cnst_ok = NB < ND;

	for (int i = 0; i < N; i++) {

		if (cnst_ok && 0 == tstrs[i][NB]) {

			if (MD_IS_SET(io, i))  {

				cnst_ok = false;
				break;
			}

			cnst_flags = MD_SET(cnst_flags, i);

			for (int d = NB; d < ND; d++)
				cnst_ok &= (0 == tstrs[i][d]);
		}
	}

	long cnst_size = 1;
	int cnst_dims = NB;

	long tsizes[N];
	for (int i = 0; i < N; i++)
		tsizes[i] = (long)sizes[i] * md_calc_size(NB, tdims);

	for (; cnst_dims < ND; cnst_dims++) {

		cnst_size *= tdims[cnst_dims];

		for (int i = 0; i < N; i++) {
			if (!gpu && cnst_size * tsizes[i] > 4096) {	// buffer too big

				cnst_size /= tdims[cnst_dims];
				cnst_dims--;
				goto out;
			}
		}
	}
out:

	if ((0 == cnst_size) || (1 > cnst_dims))
		cnst_ok = false;


	if (cnst_ok) {

		debug_printf(DP_DEBUG4, "MD constant buffer Io: %lu Cnst: %lu Size %ld.\n", io, cnst_flags, cnst_size);

		for (int i = 0; i < N; i++) {

			if (MD_IS_SET(cnst_flags, i)) {

				for (int d = NB; d < cnst_dims; d++)
					tstrs[i][d] = (0 < d) ? tdims[d - 1] * tstrs[i][d - 1] : (long)sizes[i];

				cnst_buf[i] = md_alloc_sameplace(1, MD_DIMS(cnst_size), (size_t)tsizes[i], nptr1[i]);

#ifdef USE_CUDA
				if (gpu) {
					cuda_copy_ND(1, MD_DIMS(cnst_size), MD_DIMS(tsizes[i]), cnst_buf[i], MD_DIMS(0), nptr1[i], (size_t)tsizes[i]);
				} else
#endif			
					for (long n = 0; n < cnst_size; n++)
						memcpy(cnst_buf[i]  + n * tsizes[i], nptr1[i], (size_t)tsizes[i]);
				
				nptr1[i] = cnst_buf[i];
			}
		}
	}
#endif

	int skip = min_blockdim(N, ND, tdims, nstr1, sizes);
	unsigned long flags = 0;

	debug_printf(DP_DEBUG4, "MD-Fun. Io: %lu Input: ", io);
	debug_print_dims(DP_DEBUG4, D, dim);

#ifdef USE_CUDA
	if (num_auto_parallelize && !gpu && !one_on_gpu(N, nptr1)) {
#else
	if (num_auto_parallelize) {
#endif
		flags = dims_parallel(N, io, ND, tdims, nstr1, sizes);

		while ((0 != flags) && (ffs(flags) <= skip))
			skip--;

		flags = flags >> skip;
	}

	const long* nstr2[N];

	for (int i = 0; i < N; i++)
		nstr2[i] = *nstr1[i] + skip;

#ifdef USE_CUDA
	debug_printf(DP_DEBUG4, "This is a %s call\n.", gpu ? "gpu" : "cpu");

	__block struct nary_opt_data_s data = { md_calc_size(skip, tdims), gpu ? &gpu_ops : &cpu_ops };
#else
	__block struct nary_opt_data_s data = { md_calc_size(skip, tdims), &cpu_ops };
#endif

	debug_printf(DP_DEBUG4, "Vec: %d (%ld) Opt.: ", skip, data.size);
	debug_print_dims(DP_DEBUG4, ND, tdims);

	NESTED(void, nary_opt, (void* ptr[]))
	{
		NESTED_CALL(too, (&data, ptr));
	};

	double start = timestamp();

	md_parallel_nary(N, ND - skip, tdims + skip, flags, nstr2, nptr1, nary_opt);

	double end = timestamp();

	for (int i = 0; i < N; i++)
		md_free(cnst_buf[i]);

#pragma omp critical
	md_flp_total_time += end - start;

	debug_printf(DP_DEBUG4, "MD time: %f\n", end - start);
}

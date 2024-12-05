/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2020. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2024. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2017. Intel Corporation.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2020 Martin Uecker
 * 2019-2020 Sebastian Rosenzweig
 * 2013      Frank Ong <frankong@berkeley.edu>
 * 2017      Michael J. Anderson <michael.j.anderson@intel.com>
 * 2023      Moritz Blumenthal
 * 2023      Bernhard Rapp
 *
 * Generic operations on multi-dimensional arrays. Most functions
 * come in two flavours:
 *
 * 1. A basic version which takes the number of dimensions, an array
 * of long integers specifying the size of each dimension, the pointers
 * to the data, and the size of each element and other required parameters.
 * The data is assumed to be stored in column-major format.
 *
 * 2. An extended version which takes an array of long integers which
 * specifies the strides for each argument.
 *
 * All functions should work on CPU and GPU and md_copy can be used
 * to copy between CPU and GPU.
 */

#define _GNU_SOURCE

#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

#include <strings.h>

#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/nested.h"

#include "num/optimize.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls.h"
#include "num/gpukrnls_copy.h"
#endif

#include "num/vptr.h"
#include "num/mpi_ops.h"

#include "multind.h"



static void md_nary_int(int C, int D, const long dim[D], const long* str[C], void* ptr[C], md_nary_fun_t fun)
{
	while ((D > 0) && (1 == dim[D - 1]))
		D--;

	if (0 == D) {

		NESTED_CALL(fun, (ptr));
		return;
	}

	for (long i = 0; i < dim[D - 1]; i++) {

		void* moving_ptr[C];

		for (int j = 0; j < C; j++)
			moving_ptr[j] = ptr[j] + i * str[j][D - 1];

		md_nary_int(C, D - 1, dim, str, moving_ptr, fun);
	}
}

/**
 * Generic functions which loops over all dimensions of a set of
 * multi-dimensional arrays and calls a given function for each position.
 */
void md_nary(int C, int D, const long dim[D], const long* str[C], void* ptr[C], md_nary_fun_t fun)
{
	unsigned long block_flags = 0;

	vptr_assert_sameplace(C, ptr);

	for (int i = 0; i < C; i++)
		block_flags |= vptr_block_loop_flags(D, dim, str[i], ptr[i], 1);

	long bdim[D?:1];
	long pos[D?:1];
	void* nptr[C];

	md_select_dims(D, ~block_flags, bdim, dim);
	md_set_dims(D, pos, 0);

	do {
		bool mpi_acces = true;

		for (int i = 0; i < C; i++) {

			nptr[i] = ptr[i] + md_calc_offset(D, str[i], pos);
			mpi_acces = mpi_acces && mpi_accessible(nptr[i]);
		}

		if (!mpi_acces)
			continue;

		for (int i = 0; i < C; i++)
			nptr[i] = vptr_resolve(nptr[i]);

		md_nary_int(C, D, bdim, str, nptr, fun);

	} while (md_next(D, dim, block_flags, pos));
}


/**
 * Generic functions which loops over all dimensions of a set of
 * multi-dimensional arrays and calls a given function for each position.
 * This functions tries to parallelize over the dimensions indicated
 * with flags.
 */
void md_parallel_nary(int C, int D, const long dim[D], unsigned long flags, const long* str[C], void* ptr[C], md_nary_fun_t fun)
{
	flags = flags & md_nontriv_dims(D, dim);

	if (0 == flags) {

		md_nary(C, D, dim, str, ptr, fun);
		return;
	}

	long dimc[D];
	md_select_dims(D, ~flags, dimc, dim);

	// Collect all parallel dimensions

	long parallel_dim[D];
	md_select_dims(D, flags, parallel_dim, dim);
	long total_iterations = md_calc_size(D, parallel_dim);

#ifdef _OPENMP
	int old_threads = omp_get_max_threads();
	int outer_threads = MAX(1, MIN(old_threads, total_iterations));
	int inner_threads = MAX(1, old_threads / outer_threads);

	omp_set_num_threads(outer_threads);	
#endif

#pragma omp parallel for
	for (long i = 0; i < total_iterations; i++) {

#ifdef _OPENMP
		omp_set_num_threads(inner_threads);
#endif

		// Recover place in parallel iteration space
		long iter_i[D];
		md_unravel_index(D, iter_i, ~0UL, parallel_dim, i);

		void* moving_ptr[C];

		for (int j = 0; j < C; j++)
			moving_ptr[j] = ptr[j] + md_calc_offset(D, str[j], iter_i);

		md_nary(C, D, dimc, str, moving_ptr, fun);
	}

#ifdef _OPENMP
	omp_set_num_threads(old_threads);
#endif
}


static void md_loop_r(int D, const long dim[D], unsigned long flags, long pos[D], md_loop_fun_t fun)
{
	if (0 == D) {

		NESTED_CALL(fun, (pos));
		return;
	}

	D--;

	if (!MD_IS_SET(flags, D)) {

		for (pos[D] = 0; pos[D] < dim[D]; pos[D]++)
			md_loop_r(D, dim, flags, pos, fun);

	} else {

		md_loop_r(D, dim, flags, pos, fun);
	}
}

/**
 * Generic function which loops over all dimensions and calls a given
 * function passing the current indices as argument.
 *
 * Runs fun(data, position) for all position in dim
 *
 */
void md_parallel_loop(int D, const long _dim[static D], unsigned long flags, md_loop_fun_t fun)
{
	const long *dim = _dim;	// clang

	NESTED(void, fun2, (unsigned long flags2, long *pos))
	{
		md_loop_r(D, dim, flags2, pos, fun);
	};

	md_parallel_loop_split(D, dim, flags, fun2);
}


void md_parallel_loop_split(int D, const long dim[static D], unsigned long flags, md_loop_fun2_t fun)
{
	flags &= md_nontriv_dims(D, dim);

	long psize = 1;
	long rsize = 1;

#ifdef _OPENMP
	rsize = 4 * omp_get_max_threads();
#endif
	// reduce overhead by parallelizing less dims

	for (int i = D - 1; i >= 0; i--) {

		if (psize >= rsize)
			flags = MD_CLEAR(flags, i);

		if (MD_IS_SET(flags, i))
			psize *= dim[i];
	}

	long pdims[D];
	md_select_dims(D, flags, pdims, dim);

	long iter = md_calc_size(D, pdims);

#pragma omp parallel for
	for (long i = 0; i < iter; i++) {

		// Recover place in parallel iteration space
		long pos[D];
		md_unravel_index(D, pos, ~0UL, pdims, i);
		fun(flags, pos);
	}
}


/**
 * Generic function which loops over all dimensions and calls a given
 * function passing the current indices as argument.
 *
 * Runs fun( position ) for all position in dim
 *
 */
void md_loop(int D, const long dim[D], md_loop_fun_t fun)
{
	long pos[D];
	md_loop_r(D, dim, 0, pos, fun);
}



/**
 * Computes the next position. Returns true until last index.
 */
bool md_next(int D, const long dims[D], unsigned long flags, long pos[D])
{
	if (0 == D--)
		return false;

	if (md_next(D, dims, flags, pos))
		return true;

	if (MD_IS_SET(flags, D)) {

		assert((0 <= pos[D]) && (pos[D] < dims[D]));

		if (++pos[D] < dims[D])
			return true;

		pos[D] = 0;
	}

	return false;
}



/**
 * Returns offset for position in a multidimensional array
 *
 * return pos[0]*strides[0] + ... + pos[D-1]*strides[D-1]
 *
 * @param D number of dimensions
 * @param dim dimensions array
 */
long md_calc_offset(int D, const long strides[D], const long position[D])
{
	long pos = 0;

	for (int i = 0; i < D; i++)
		pos += strides[i] * position[i];

	return pos;
}



static long md_calc_size_r(int D, const long dim[D], size_t size)
{
	if (0 == D)
		return (long)size;

	return md_calc_size_r(D - 1, dim, (size_t)((long)size * dim[D - 1]));
}

/**
 * Returns the number of elements
 *
 * return dim[0]*dim[1]*...*dim[D-1]
 *
 * @param D number of dimensions
 * @param dim dimensions array
 */
long md_calc_size(int D, const long dim[D])
{
	return md_calc_size_r(D, dim, 1);
}



/**
 * Computes the number of smallest dimensions which are stored
 * continuously, i.e. can be accessed as a block of memory.
 *
 */
int md_calc_blockdim(int D, const long dim[D], const long str[D], size_t size)
{
	long dist = (long)size;
	int i = 0;

	for (i = 0; i < D; i++) {

		if (!((str[i] == dist) || (dim[i] == 1)))
			break;

		dist *= dim[i];
	}

	return i;
}



/**
 * Copy dimensions specified by flags and set remaining dimensions to 1
 *
 * odims = [ 1  idims[1]  idims[2]  1  1  idims[5] ]
 *
 * @param D number of dimensions
 * @param flags bitmask specifying which dimensions to copy
 * @param odims output dimensions
 * @param idims input dimensions
 */
void md_select_dims(int D, unsigned long flags, long odims[D], const long idims[D])
{
	for (int i = 0; i < D; i++)
		if (!MD_IS_SET(flags, i))
			odims[i] = 1;
		else
			odims[i] = idims[i];
}

/**
 * Copy strides specified by flags and set remaining dimensions to 0
 *
 * ostrs = [ 0  istrs[1]  istrs[2]  0  0  istrs[5] ]
 *
 * @param D number of dimensions
 * @param flags bitmask specifying which dimensions to copy
 * @param ostrs output strides
 * @param istrs input strides
 */
void md_select_strides(int D, unsigned long flags, long ostrs[D], const long istrs[D])
{
       md_copy_dims(D, ostrs, istrs);

       for (int i = 0; i < D; i++)
               if (!MD_IS_SET(flags, i))
                       ostrs[i] = 0;
}

/**
 * Copy dimensions
 *
 * odims[i] = idims[i]
 */
void md_copy_dims(int D, long odims[D], const long idims[D])
{
	memcpy(odims, idims, (size_t)(D * (long)sizeof(long)));
}



/**
 * Copy strides
 *
 * ostrs[i] = istrs[i]
 */
void md_copy_strides(int D, long ostrs[D], const long istrs[D])
{
	memcpy(ostrs, istrs, (size_t)(D  * (long)sizeof(long)));
}



/**
 * Set all dimensions to value
 *
 * dims[i] = val
 */
void md_set_dims(int D, long dims[D], long val)
{
	for (int i = 0; i < D; i++)
		dims[i] = val;
}



/**
 * returns whether or not @param pos is a valid index of an array of dimension @param dims
 */
bool md_is_index(int D, const long pos[D], const long dims[D])
{
	if (D == 0)
		return true;

	return ((pos[0] >= 0) && (pos[0] < dims[0]) && md_is_index(D - 1, pos + 1, dims + 1));
}



/**
 * return whether some other dimensions are >1
 */
bool md_check_dimensions(int N, const long dims[N], unsigned long flags)
{
	long d[N];
	md_select_dims(N, ~flags, d, dims);

	return (1 != md_calc_size(N, d));
}



/**
 * Check if dimensions at 'flags' position are equal
 */
bool md_check_equal_dims(int N, const long dims1[N], const long dims2[N], unsigned long flags)
{
	return (   md_check_bounds(N, flags, dims1, dims2)
	        && md_check_bounds(N, flags, dims2, dims1));
}



/*
 * compute non-trivial (> 1) dims
 */
unsigned long md_nontriv_dims(int D, const long dims[D])
{
	unsigned long flags = 0;

	for (int i = 0; i < D; i++)
		if (dims[i] > 1)
			flags = MD_SET(flags, i);

	return flags;
}


/*
 * compute non-trivial (!= 0) strides
 */
unsigned long md_nontriv_strides(int D, const long strs[D])
{
	unsigned long flags = 0;

	for (int i = 0; i < D; i++)
		if (strs[i] != 0)
			flags = MD_SET(flags, i);

	return flags;
}



/**
 * Set all dimensions to one
 *
 * dims[i] = 1
 */
void md_singleton_dims(int D, long dims[D])
{
	for (int i = 0; i < D; i++)
		dims[i] = 1;
}



/**
 * Set all strides to one
 *
 * dims[i] = 1
 */
void md_singleton_strides(int D, long strs[D])
{
	for (int i = 0; i < D; i++)
		strs[i] = 0;
}



/**
 * Check dimensions for compatibility. Dimensions must be equal or
 * where indicated by a set bit in flags one must be equal to one
 * in at least one of the arguments.
 */
bool md_check_compat(int D, unsigned long flags, const long dim1[D], const long dim2[D])
{
	if (0 == D)
		return true;

	D--;

	if ((dim1[D] == dim2[D]) || (MD_IS_SET(flags, D) && ((1 == dim1[D]) || (1 == dim2[D]))))
		return md_check_compat(D, flags, dim1, dim2);

	return false;
}



void md_merge_dims(int N, long out_dims[N], const long dims1[N], const long dims2[N])
{
	assert(md_check_compat(N, ~0UL, dims1, dims2));

	for (int i = 0; i < N; i++)
		out_dims[i] = (1 == dims1[i]) ? dims2[i] : dims1[i];
}



/**
 * dim1 must be bounded by dim2 where a bit is set
 */
bool md_check_bounds(int D, unsigned long flags, const long dim1[D], const long dim2[D])
{
	if (0 == D--)
		return true;

	if (!MD_IS_SET(flags, D) || (dim1[D] <= dim2[D]))
		return md_check_bounds(D, flags, dim1, dim2);

	return false;
}


/**
 * Set the output's flagged dimensions to the minimum of the two input dimensions
 *
 * odims = [ MIN(idims1[0],idims2[0] ... MIN(idims1[D-1],idims2[D-1]) ]
 *
 * @param D number of dimensions
 * @param flags bitmask specifying which dimensions to minimize
 * @param odims output dimensions
 * @param idims1 input 1 dimensions
 * @param idims2 input 2 dimensions
 */
void md_min_dims(int D, unsigned long flags, long odims[D], const long idims1[D], const long idims2[D])
{
	for (int i = 0; i < D; i++)
		if (MD_IS_SET(flags, i))
			odims[i] = MIN(idims1[i], idims2[i]);
}


/**
 * Set the output's flagged dimensions to the maximum of the two input dimensions
 *
 * odims = [ MAX(idims1[0],idims2[0] ... MAX(idims1[D-1],idims2[D-1]) ]
 *
 * @param D number of dimensions
 * @param flags bitmask specifying which dimensions to maximize
 * @param odims output dimensions
 * @param idims1 input 1 dimensions
 * @param idims2 input 2 dimensions
 */
void md_max_dims(int D, unsigned long flags, long odims[D], const long idims1[D], const long idims2[D])
{
	for (int i = 0; i < D; i++)
		if (MD_IS_SET(flags, i))
			odims[i] = MAX(idims1[i], idims2[i]);
}



/**
 * Zero out array (with strides)
 *
 * ptr[i] = 0
 */
void md_clear2(int D, const long dim[D], const long str[D], void* ptr, size_t size)
{
	const long (*nstr[1])[D] = { (const long (*)[D])str };
#ifdef	USE_CUDA
	bool use_gpu = cuda_ondevice(ptr);
#endif
	unsigned long flags = 0;

	for (int i = 0; i < D; i++)
		if (0 == str[i])
			flags |= MD_BIT(i);

	long dim2[D];
	md_select_dims(D, ~flags, dim2, dim);


	NESTED(void, nary_clear, (struct nary_opt_data_s* opt_data, void* ptr[]))
	{
		size_t size2 = (size_t)((long)size * opt_data->size);

#ifdef 	USE_CUDA
		if (use_gpu) {

			cuda_clear((long)size2, ptr[0]);
			return;
		}
#endif
		memset(ptr[0], 0, size2);
	};

	optimized_nop(1, MD_BIT(0), D, dim2, nstr, (void*[1]){ ptr }, (size_t[1]){ size }, nary_clear);
}



/**
 * Calculate strides in column-major format
 * (smallest index is sequential)
 *
 * @param D number of dimensions
 * @param array of calculates strides
 * @param dim array of dimensions
 * @param size of a single element
 */
long* md_calc_strides(int D, long str[D], const long dim[D], size_t size)
{
	long old = (long)size;

	for (int i = 0; i < D; i++) {

		str[i] = (1 == dim[i]) ? 0 : old;
		old *= dim[i];
	}

	return str;
}



/**
 * Zero out array (without strides)
 *
 * ptr[i] = 0
 *
 * @param D number of dimensions
 * @param dim dimensions array
 * @param ptr pointer to data to clear
 * @param size sizeof()
 */
void md_clear(int D, const long dim[D], void* ptr, size_t size)
{
	md_clear2(D, dim, MD_STRIDES(D, dim, size), ptr, size);
}




/**
 * Copy array (with strides)
 *
 * optr[i] = iptr[i]
 */
void md_copy2(int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
{
#if 0
	// this is for a fun comparison between our copy engine and FFTW

	extern void fft2(unsigned int D, const long dim[D], unsigned int flags,
			const long ostr[D], void* optr, const long istr[D], const void* iptr);

	if (sizeof(complex float) == size)
		fft2(D, dim, 0, ostr, optr, istr, iptr);
#endif
	if (0 == md_calc_size(D, dim))
		return;

	if (is_mpi(optr) || is_mpi(iptr)) {

//		debug_print_dims(DP_INFO, D, dim);
//		debug_print_dims(DP_INFO, D, ostr);
//		debug_print_dims(DP_INFO, D, istr);

		unsigned long iflags = vptr_block_loop_flags(D, dim, istr, iptr, size);
		unsigned long oflags = vptr_block_loop_flags(D, dim, ostr, optr, size);

		long cdims[D];
		md_singleton_dims(D, cdims);

		long ldims[D];
		md_copy_dims(D, ldims, dim);

		for (int i = 0; i < D; i++) {

			if (MD_IS_SET(iflags, i) || MD_IS_SET(oflags, i))
				break;

			if (1 == ldims[i])
				continue;

			if ((ostr[i] == (long)size) && (istr[i] == (long)size)) {

				size = (size_t)((long)size * ldims[i]);
				ldims[i] = 1;
			}
		}

		long pos[D];
		md_set_dims(D, pos, 0);

		do {
			void* dst = optr + md_calc_offset(D, ostr, pos);
			const void* src = iptr + md_calc_offset(D, istr, pos);


			if (!is_mpi(src)) {

				if (mpi_accessible(dst)) {

					dst = vptr_resolve(dst);
					src = vptr_resolve(src);
					md_copy(D, cdims, dst, src, size);
				}
				continue;
			}

			if (!is_mpi(dst)) {

				int root = mpi_ptr_get_rank(src);

				dst = vptr_resolve_unchecked(dst);

				if (mpi_accessible(src)) {

					src = vptr_resolve_unchecked(src);
					md_copy(D, cdims, dst, src, size);
				}

				if (-1 < root)
					mpi_bcast(dst, (long)size, root);

				continue;
			}

			for (int receiver = 0; receiver < mpi_get_num_procs(); receiver++) {

				if (!mpi_accessible_from(dst, receiver))
					continue;

				int sender = mpi_accessible_from(src, receiver) ? receiver : mpi_ptr_get_rank(src);

				const void* _src = (mpi_get_rank() == sender) ? vptr_resolve(src) : NULL;
				void* _dst = (mpi_get_rank() == receiver) ? vptr_resolve(dst) : NULL;

				mpi_copy(_dst, (long)size, _src, sender, receiver);
			}

		} while (md_next(D, ldims, ~0UL, pos));

		return;
	}

	iptr = vptr_resolve(iptr);
	optr = vptr_resolve(optr);

#ifdef	USE_CUDA
	bool use_gpu = cuda_ondevice(optr) || cuda_ondevice(iptr);

#if 1
	long tostr[D];
	long tistr[D];
	long tdims[D];

	md_copy_strides(D, tostr, ostr);
	md_copy_strides(D, tistr, istr);
	md_copy_dims(D, tdims, dim);

	long (*nstr2[2])[D] = { &tostr, &tistr };
	int ND = optimize_dims_gpu(2, D, tdims, nstr2);

	assert(ND <= D);

#if 1
	// permute dims with 0 input strides or negative in/output strides to the end
	// these might be permuted to the inner dimensions by optimize_dims and break the strided copy

	int perm[ND];

	for (int i = 0, j = 0; i < ND; i++) {

		if (   (0 >= (*nstr2[1])[i])
		    || (0 >= (*nstr2[0])[i])) {

			perm[ND - 1 -j] = i;
			j += 1;

		} else {

			perm[i - j] = i;
		}
	}

	long tmp[ND];

	md_permute_dims(ND, perm, tmp, tdims);
	md_copy_dims(ND, tdims, tmp);

	md_permute_dims(ND, perm, tmp, tostr);
	md_copy_dims(ND, tostr, tmp);

	md_permute_dims(ND, perm, tmp, tistr);
	md_copy_dims(ND, tistr, tmp);
#endif

#ifdef USE_CUDA
	if (use_gpu && (cuda_ondevice(optr) == cuda_ondevice(iptr)) && ND <= 7) {

		cuda_copy_ND(ND, tdims, tostr, optr, tistr, iptr, size);
		return;
	}
#endif

#if 1
	//fill like copies

	unsigned long fill_flags =  md_nontriv_dims(D, tdims)
				 & ~md_nontriv_strides(D, tistr)
				 & md_nontriv_strides(D, tostr);

	if (use_gpu && (0 != fill_flags)) {

		int idx = md_min_idx(fill_flags);

		long tdims2[ND];
		long pos[ND];

		md_select_dims(ND, ~MD_BIT(idx), tdims2, tdims);
		md_singleton_strides(ND, pos);

		md_copy2(ND, tdims2, tostr, optr, tistr, iptr, size);

		pos[idx] = 1;

		while (pos[idx] < tdims[idx]) {

			tdims2[idx] = MIN(pos[idx], tdims[idx] - pos[idx]);

			md_copy2(ND, tdims2, tostr, optr + md_calc_offset(ND, tostr, pos), tostr, optr, size);

			pos[idx] += tdims2[idx];
		}

		return;
	}
#endif

	size_t sizes[2] = { size, size };
	int skip = min_blockdim(2, ND, tdims, nstr2, sizes);

	debug_printf(DP_DEBUG4, "md_copy_2 skip=%d\n", skip);
	debug_print_dims(DP_DEBUG4, ND, tdims);
	debug_print_dims(DP_DEBUG4, ND, (*nstr2[0]));
	debug_print_dims(DP_DEBUG4, ND, (*nstr2[1]));

	if (   use_gpu
	    && (ND - skip > 0)) {

		assert(skip < ND);

		long ostr2 = (*nstr2[0])[skip];
		long istr2 = (*nstr2[1])[skip];

		if (!(   (ostr2 > 0)
	              && (istr2 > 0)))
			goto out;

		void* nptr[2] = { optr, (void*)iptr };
		long sizes[2] = { md_calc_size(skip, tdims) * (long)size, tdims[skip] };

		skip++;

		const long* nstr[2] = { *nstr2[0] + skip, *nstr2[1] + skip };

		long* sizesp = sizes; // because of clang
		void** nptrp = nptr;

		NESTED(void, nary_strided_copy, (void* ptr[]))
		{
			debug_printf(DP_DEBUG4, "CUDA 2D copy %ld %ld %ld %ld %ld %ld\n",
				sizesp[0], sizesp[1], ostr2, istr2, nptrp[0], nptrp[1]);

			cuda_memcpy_strided(sizesp, ostr2, ptr[0], istr2, ptr[1]);
		};

		md_nary(2, ND - skip, tdims + skip, nstr, nptr, nary_strided_copy);
		return;
	}

out:	;
#endif
#endif
	const long (*nstr[2])[D] = { (const long (*)[D])ostr, (const long (*)[D])istr };

	NESTED(void, nary_copy, (struct nary_opt_data_s* opt_data, void* ptr[]))
	{
		size_t size2 = (size_t)((long)size * opt_data->size);

#ifdef  USE_CUDA
		if (use_gpu) {

			cuda_memcpy((long)size2, ptr[0], ptr[1]);
			return;
		}
#endif

		memcpy(ptr[0], ptr[1], size2);
	};

	optimized_nop(2, MD_BIT(0), D, dim, nstr, (void*[2]){ optr, (void*)iptr }, (size_t[2]){ size, size }, nary_copy);
}



/**
 * Copy array (without strides)
 *
 * optr[i] = iptr[i]
 */
void md_copy(int D, const long dim[D], void* optr, const void* iptr, size_t size)
{
	long str[D];
	md_calc_strides(D, str, dim, size);

	md_copy2(D, dim, str, optr, str, iptr, size);
}



#ifdef USE_CUDA
// copied from flpmath.c
static void* gpu_constant(const void* vp, size_t size)
{
        return md_gpu_move(1, (long[1]){ 1 }, vp, size);
}
#endif

/**
 * Fill array with value pointed by pointer (with strides)
 *
 * ptr[i] = iptr[0]
 */
void md_fill2(int D, const long dim[D], const long str[D], void* ptr, const void* iptr, size_t size)
{
#ifdef USE_CUDA
	if (cuda_ondevice(ptr) && (!cuda_ondevice(iptr))) {

		void* giptr = gpu_constant(iptr, size);

		md_fill2(D, dim, str, ptr, giptr, size);

		md_free(giptr);
		return;
	}
#endif

	long istr[D];
	md_singleton_strides(D, istr);

	md_copy2(D, dim, str, ptr, istr, iptr, size);
}



/**
 * Fill array with value pointed by pointer (without strides)
 *
 * ptr[i] = iptr[0]
 */
void md_fill(int D, const long dim[D], void* ptr, const void* iptr, size_t size)
{
	md_fill2(D, dim, MD_STRIDES(D, dim, size), ptr, iptr, size);
}




/**
 * Swap values between a number of arrays (with strides)
 */
void md_circular_swap2(int M, int D, const long dims[D], const long* strs[M], void* ptr[M], size_t size)
{
	size_t sizes[M];

	for (int i = 0; i < M; i++)
		sizes[i] = size;

	const long (*nstrs[M])[D];

	for (int i = 0; i < M; i++)
		nstrs[i] = (const long (*)[D])strs[i];


	NESTED(void, nary_swap, (struct nary_opt_data_s* opt_data, void* ptr[]))
	{
		size_t size2 = (size_t)((long)size * opt_data->size);

		char* tmp = (size2 < 32) ? alloca(size2) : xmalloc(size2);

#ifdef  USE_CUDA
		assert(!cuda_ondevice(ptr[0]));
		assert(!cuda_ondevice(ptr[1]));
#endif
		memcpy(tmp, ptr[0], size2);

		for (int i = 0; i < M - 1; i++)
			memcpy(ptr[i], ptr[i + 1], size2);

		memcpy(ptr[M - 1], tmp, size2);

		if (size2 >= 32)
			xfree(tmp);
	};

	optimized_nop(M, (1UL << M) - 1, D, dims, nstrs, ptr, sizes, nary_swap);
}



/**
 * Swap values between a number of arrays
 */
void md_circular_swap(int M, int D, const long dims[D], void* ptr[M], size_t size)
{
	long strs[M][D];

	md_calc_strides(D, strs[0], dims, size);

	const long* strp[M];

	strp[0] = strs[0];

	for (int i = 1; i < M; i++) {

		md_copy_strides(D, strs[i], strs[0]);
		strp[i] = strs[i];
	}

	md_circular_swap2(M, D, dims, strp, ptr, size);
}



/**
 * Swap values between two arrays (with strides)
 *
 * iptr[i] = optr[i] and optr[i] = iptr[i]
 */
void md_swap2(int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], void* iptr, size_t size)
{
	md_circular_swap2(2, D, dim, (const long*[2]){ ostr, istr }, (void*[2]){ optr, iptr }, size);
}



/**
 * Swap values between two arrays (without strides)
 *
 * iptr[i] = optr[i] and optr[i] = iptr[i]
 */
void md_swap(int D, const long dim[D], void* optr, void* iptr, size_t size)
{
	long str[D];
	md_calc_strides(D, str, dim, size);

	md_swap2(D, dim, str, optr, str, iptr, size);
}



/**
 * Move a block from an array to another array (with strides)
 *
 */
void md_move_block2(int D, const long dim[D], const long opos[D], const long odim[D], const long ostr[D], void* optr, const long ipos[D], const long idim[D], const long istr[D], const void* iptr, size_t size)
{
	for (int i = 0; i < D; i++) {

		assert(dim[i] <= odim[i]);
		assert(dim[i] <= idim[i]);
		assert((0 <= opos[i]) && (opos[i] <= odim[i] - dim[i]));
		assert((0 <= ipos[i]) && (ipos[i] <= idim[i] - dim[i]));
	}

	long ioff = md_calc_offset(D, istr, ipos);
	long ooff = md_calc_offset(D, ostr, opos);

	md_copy2(D, dim, ostr, optr + ooff, istr, iptr + ioff, size);
}


/**
 * Move a block from an array to another array (without strides)
 *
 */
void md_move_block(int D, const long dim[D], const long opos[D], const long odim[D], void* optr, const long ipos[D], const long idim[D], const void* iptr, size_t size)
{
	md_move_block2(D, dim,
			opos, odim, MD_STRIDES(D, odim, size), optr,
			ipos, idim, MD_STRIDES(D, idim, size), iptr, size);
}


/**
 * Copy a block from an array to another array (with strides)
 *
 * Block dimensions are min(idim , odim)
 *
 * if idim[d] > odim[d], then optr[i] = iptr[pos + i] for 0 <= i < odim[d]
 *
 * if idim[d] < odim[d], then optr[pos + i] = iptr[i] for 0 <= i < idim[d]
 *
 */
void md_copy_block2(int D, const long pos[D], const long odim[D], const long ostr[D], void* optr, const long idim[D], const long istr[D], const void* iptr, size_t size)
{
	long dim[D];
	long ipos[D];
	long opos[D];

	for (int i = 0; i < D; i++) {

		assert((idim[i] != odim[i]) || (0 == pos[i]));

		dim[i] = MIN(odim[i], idim[i]);
		ipos[i] = 0;
		opos[i] = 0;

		if (idim[i] != dim[i])
			ipos[i] = pos[i];

		if (odim[i] != dim[i])
			opos[i] = pos[i];
	}

	md_move_block2(D, dim, opos, odim, ostr, optr, ipos, idim, istr, iptr, size);
}



/**
 * Copy a block from an array to another array (without strides)
 *
 * Block dimensions are min(idim , odim)
 *
 * if idim[d] > odim[d], then optr[i] = iptr[pos + i] for 0 <= i < odim[d]
 *
 * if idim[d] < odim[d], then optr[pos + i] = iptr[i] for 0 <= i < idim[d]
 *
 */
void md_copy_block(int D, const long pos[D], const long odim[D], void* optr, const long idim[D], const void* iptr, size_t size)
{
	md_copy_block2(D, pos,
			odim, MD_STRIDES(D, odim, size), optr,
			idim, MD_STRIDES(D, idim, size), iptr, size);
}



/**
 * Resize an array by zero-padding or by truncation at the end.
 *
 * optr = [iptr 0 0 0 0]
 *
 */
void md_resize(int D, const long odim[D], void* optr, const long idim[D], const void* iptr, size_t size)
{
	long pos[D];
	memset(pos, 0, (size_t)(D * (long)sizeof(long)));

	for (int i = 0; i < D; i++) {

		if (odim[i] > idim[i]) {

			md_clear(D, odim, optr, size);
			break;
		}
	}

	md_copy_block(D, pos, odim, optr, idim, iptr, size);
}

/**
 * Pad an array by val at the end.
 *
 * optr = [iptr val val val val]
 *
 */
void md_pad(int D, const void* val, const long odim[D], void* optr, const long idim[D], const void* iptr, size_t size)
{
	long pos[D];
	memset(pos, 0, (size_t)(D * (long)sizeof(long)));

	md_fill(D, odim, optr, val, size);
	md_copy_block(D, pos, odim, optr, idim, iptr, size);
}

/**
 * Resize an array by zero-padding or by truncation at both ends symmetrically.
 *
 * optr = [0 0 iptr 0 0]
 *
 */
void md_resize_center(int D, const long odim[D], void* optr, const long idim[D], const void* iptr, size_t size)
{
	// the definition of the center position corresponds
	// to the one used in the FFT.

	long pos[D];
	for (int i = 0; i < D; i++)
		pos[i] = labs((odim[i] / 2) - (idim[i] / 2));

	for (int i = 0; i < D; i++) {

		if (odim[i] <= idim[i])
			continue;

		md_clear(D, odim, optr, size);
		break;
	}

	md_copy_block(D, pos, odim, optr, idim, iptr, size);
}



/**
 * Resize an array by zero-padding or by truncation at the beginning.
 *
 * optr = [0 0 0 0 iptr]
 *
 */
void md_resize_front(int D, const long odim[D], void* optr, const long idim[D], const void* iptr, size_t size)
{
	long pos[D];
	for (int i = 0; i < D; i++)
		pos[i] = labs(odim[i] - idim[i]);

	for (int i = 0; i < D; i++) {

		if (odim[i] <= idim[i])
			continue;

		md_clear(D, odim, optr, size);
		break;
	}

	md_copy_block(D, pos, odim, optr, idim, iptr, size);
}


/**
 * Pad an array on both ends by val.
 *
 * optr = [val val iptr val val]
 *
 */
void md_pad_center(int D, const void* val, const long odim[D], void* optr, const long idim[D], const void* iptr, size_t size)
{
	long pos[D];

	for (int i = 0; i < D; i++)
		pos[i] = labs((odim[i] / 2) - (idim[i] / 2));

	md_fill(D, odim, optr, val, size);
	md_copy_block(D, pos, odim, optr, idim, iptr, size);
}


void md_reflectpad_center2(int D, const long odim[D], const long ostr[D], void* optr,
			const long idim[D], const long istr[D], const void* iptr, size_t size)
{
	long odim2[D];
	long ristr[D];
	long loop_idx[D];
	long blockdim[D];
	long center_block[D];
	long block0_size[D];
	long count = 0;

	for (int i = 0; i < D; i++) {

		assert(odim[i] >= idim[i]);

		blockdim[i] = 1;
		center_block[i] = 0;

		ristr[i] = istr[i];
		odim2[i] = idim[i];

		block0_size[i] = 0;

		if (odim[i] > idim[i]) {

			loop_idx[count++] = i;

			long main_start = labs((odim[i] / 2) - (idim[i] / 2));
			long main_end = main_start + idim[i];
			long before = (main_start + idim[i] - 1) / idim[i];
			long after = (odim[i] - main_end + idim[i] - 1) / idim[i];

			blockdim[i] = 1 + before + after;
			center_block[i] = before;

			long x = main_start % idim[i];

			block0_size[i] = (0 == x) ? idim[i] : x;
		}
	}

	long block_pos[D];
	long in_pos[D];

	md_set_dims(D, block_pos, 0);
	md_set_dims(D, in_pos, 0);

	long opos[D];
	md_set_dims(D, opos, 0);

	do {
		for (int i = 0, idx = loop_idx[0]; i < count; idx = (++i < count) ? loop_idx[i] : idx) {

			opos[idx] = (block_pos[idx] >= 1) ? (block0_size[idx] + idim[idx] * (block_pos[idx] - 1)) : 0;
			odim2[idx] = (block_pos[idx] == 0) ? block0_size[idx] : MIN(idim[idx], odim[idx] - opos[idx]);

			if (1 == labs(center_block[idx] - block_pos[idx]) % 2) {

				ristr[idx] = -istr[idx];
				in_pos[idx] = (odim2[idx] < idim[idx]) ? ((block_pos[idx] > center_block[idx]) ? (idim[idx] - 1) : odim2[idx] - 1) : (idim[idx] - 1);

			} else {

				ristr[idx] = istr[idx];
				in_pos[idx] = (odim2[idx] < idim[idx]) ? ((block_pos[idx] > center_block[idx]) ? 0 : (idim[idx] - odim2[idx])) : 0;
			}
		}

		md_copy2(D, odim2, ostr, md_calc_offset(D, ostr, opos) + optr, ristr, md_calc_offset(D, istr, in_pos) + iptr, size);

	} while (md_next(D, blockdim, ~0U, block_pos));
}

void md_reflectpad_center(int D, const long odim[D], void* optr, const long idim[D], const void* iptr, size_t size)
{
	md_reflectpad_center2(D, odim, MD_STRIDES(D, odim, size), optr,
				idim, MD_STRIDES(D, idim, size), iptr, size);
}


/**
 * Extract slice from array specified by flags (with strides)
 *
 * optr = iptr(pos[0], :, pos[2], :, :)
 *
 */
void md_slice2(int D, unsigned long flags, const long pos[D], const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
{
	long odim[D];
	md_select_dims(D, ~flags, odim, dim);

	md_copy_block2(D, pos, odim, ostr, optr, dim, istr, iptr, size);
}



/**
 * Extract slice from array specified by flags (with strides)
 *
 * optr = iptr(pos[0], :, pos[2], :, :)
 *
 */
void md_slice(int D, unsigned long flags, const long pos[D], const long dim[D], void* optr, const void* iptr, size_t size)
{
	long odim[D];
	md_select_dims(D, ~flags, odim, dim);

	md_slice2(D, flags, pos, dim,
			MD_STRIDES(D, odim, size), optr,
			MD_STRIDES(D, dim, size), iptr, size);
}



/**
 * Permute array (with strides)
 *
 * optr[order[i]] = iptr[i]
 *
 */
void md_permute2(int D, const int order[D], const long odims[D], const long ostr[D], void* optr, const long idims[D], const long istr[D], const void* iptr, size_t size)
{
	unsigned long flags = 0;
	long ostr2[D];

	for (int i = 0; i < D; i++) {

		assert(order[i] < D);
		assert(odims[i] == idims[order[i]]);

		flags = MD_SET(flags, order[i]);

		ostr2[order[i]] = ostr[i];
	}

	assert(MD_BIT(D) == flags + 1U);

	md_copy2(D, idims, ostr2, optr, istr, iptr, size);
}



/**
 * Permute array (without strides)
 *
 * optr[order[i]] = iptr[i]
 *
 */
void md_permute(int D, const int order[D], const long odims[D], void* optr, const long idims[D], const void* iptr, size_t size)
{
	md_permute2(D, order,
			odims, MD_STRIDES(D, odims, size), optr,
			idims, MD_STRIDES(D, idims, size), iptr, size);
}



/**
 * Permute dimensions
 *
 *
 */
void md_permute_dims(int D, const int order[D], long odims[D], const long idims[D])
{
	for (int i = 0; i < D; i++)
		odims[i] = idims[order[i]];
}



static void md_transpose_order(int D, int order[D], int dim1, int dim2)
{
	assert(dim1 < D);
	assert(dim2 < D);

	for (int i = 0; i < D; i++)
		order[i] = i;

	order[dim1] = dim2;
	order[dim2] = dim1;
}

/**
 * Transpose dimensions
 *
 *
 */
void md_transpose_dims(int D, int dim1, int dim2, long odims[D], const long idims[D])
{
	int order[D];
	md_transpose_order(D, order, dim1, dim2);

	md_permute_dims(D, order, odims, idims);
}



/**
 * Transpose array (with strides)
 *
 * optr[dim2] = iptr[dim1]
 *
 * optr[dim1] = iptr[dim2]
 *
 */
void md_transpose2(int D, int dim1, int dim2, const long odims[D], const long ostr[D], void* optr, const long idims[D], const long istr[D], const void* iptr, size_t size)
{
	for (int i = 0; i < D; i++)
		if ((i != dim1) && (i != dim2))
			assert(odims[i] == idims[i]);

	assert(odims[dim1] == idims[dim2]);
	assert(odims[dim2] == idims[dim1]);

	int order[D];
	md_transpose_order(D, order, dim1, dim2);

	md_permute2(D, order, odims, ostr, optr, idims, istr, iptr, size);
}



/**
 * Transpose array (without strides)
 *
 * optr[dim2] = iptr[dim1]
 *
 * optr[dim1] = iptr[dim2]
 *
 */
void md_transpose(int D, int dim1, int dim2, const long odims[D], void* optr, const long idims[D], const void* iptr, size_t size)
{
	md_transpose2(D, dim1, dim2,
			odims, MD_STRIDES(D, odims, size), optr,
			idims, MD_STRIDES(D, idims, size), iptr, size);
}



static void md_flip_inpl2(int D, const long dims[D], unsigned long flags, const long str[D], void* ptr, size_t size);

/**
 * Swap input and output while flipping selected dimensions
 * at the same time.
 */
void md_swap_flip2(int D, const long dims[D], unsigned long flags, const long ostr[D], void* optr, const long istr[D], void* iptr, size_t size)
{
#if 1
	int i;

	for (i = D - 1; i >= 0; i--)
		if ((1 != dims[i]) && MD_IS_SET(flags, i))
			break;

	if (-1 == i) {

		md_swap2(D, dims, ostr, optr, istr, iptr, size);
		return;
	}

	assert(1 < dims[i]);
	assert(ostr[i] != 0);
	assert(istr[i] != 0);

	long dims2[D];

	md_copy_dims(D, dims2, dims);

	dims2[i] = dims[i] / 2;

	long off = (dims[i] + 1) / 2;

	assert(dims2[i] + off == dims[i]);

	md_swap_flip2(D, dims2, flags, ostr, optr, istr, iptr + off * istr[i], size);
	md_swap_flip2(D, dims2, flags, ostr, optr + off * ostr[i], istr, iptr, size);

	// odd, swap center plane
	// (we should split in three similar sized chunks instead)

	dims2[i] = 1;

	if (1 == dims[i] % 2)
		md_swap_flip2(D, dims2, flags, ostr, optr + (off - 1) * ostr[i], istr, iptr + (off - 1) * istr[i], size);
#else
	// simpler, but more swaps

	md_swap2(D, dims, ostr, optr, istr, iptr, size);
	md_flip_inpl2(D, dims, flags, ostr, optr, size);
	md_flip_inpl2(D, dims, flags, istr, iptr, size);
#endif
}



/**
 * Swap input and output while flipping selected dimensions
 * at the same time.
 */
void md_swap_flip(int D, const long dims[D], unsigned long flags, void* optr, void* iptr, size_t size)
{
	long strs[D];
	md_calc_strides(D, strs, dims, size);

	md_swap_flip2(D, dims, flags, strs, optr, strs, iptr, size);
}



static void md_flip_inpl2(int D, const long dims[D], unsigned long flags, const long str[D], void* ptr, size_t size)
{
	int i;

	assert(0 == (vptr_block_loop_flags(D, dims, str, ptr, size) & flags));

	for (i = D - 1; i >= 0; i--)
		if ((1 != dims[i]) && MD_IS_SET(flags, i))
			break;

	if (-1 == i)
		return;

	assert(1 < dims[i]);
	assert(str[i] != 0);

	long dims2[D];

	md_copy_dims(D, dims2, dims);

	dims2[i] = dims[i] / 2;

	long off = str[i] * (0 + (dims[i] + 1) / 2);

	md_swap_flip2(D, dims2, flags, str, ptr, str, ptr + off, size);
}

/**
 * Flip array (with strides)
 *
 * optr[dims[D] - 1 - i] = iptr[i]
 *
 */
void md_flip2(int D, const long dims[D], unsigned long flags, const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
{
	if (optr == iptr) {

		assert(ostr == istr);

		md_flip_inpl2(D, dims, flags, ostr, optr, size);
		return;
	}

	long off = 0;
	long ostr2[D];

	for (int i = 0; i < D; i++) {

		ostr2[i] = ostr[i];

		if (MD_IS_SET(flags, i)) {

			ostr2[i] = -ostr[i];
			off += (dims[i] - 1) * ostr[i];
		}
	}

	md_copy2(D, dims, ostr2, optr + off, istr, iptr, size);
}



/**
 * Flip array (without strides)
 *
 * optr[dims[D] - 1 - i] = iptr[i]
 *
 */
void md_flip(int D, const long dims[D], unsigned long flags, void* optr, const void* iptr, size_t size)
{
	long str[D];
	md_calc_strides(D, str, dims, size);

	md_flip2(D, dims, flags, str, optr, str, iptr, size);
}


/**
 * Reshape array (with strides)
 *
 * Only flagged dims may flow
 */
void md_reshape2(int D, unsigned long flags, const long odims[D], const long ostrs[D], void* optr, const long idims[D], const long istrs[D], const void* iptr, size_t size)
{
	assert(md_calc_size(D, odims) == md_calc_size(D, idims));
	assert(md_check_equal_dims(D, odims, idims, ~flags));

	int order[D];
	int j = 0;

	for (int i = 0; i < D; i++)
		if (MD_IS_SET(flags, i))
			order[j++] = i;

	for (int i = 0; i < D; i++)
		if (!MD_IS_SET(flags, i))
			order[j++] = i;

	assert(D == j);


	int iorder[D];

	for (int i = 0; i < D; i++)
		iorder[order[i]] = i;


	long dims2[D];
	long strs2[D];

	// FIXME: we could avoid the buffer in some cases

	void* buf = md_alloc_sameplace(D, odims, size, optr);


	md_permute_dims(D, order, dims2, idims);
	md_calc_strides(D, strs2, dims2, size);

	md_permute2(D, order, dims2, strs2, buf, idims, istrs, iptr, size);


	md_permute_dims(D, order, dims2, odims);
	md_calc_strides(D, strs2, dims2, size);

	md_permute2(D, iorder, odims, ostrs, optr, dims2, strs2, buf, size);


	md_free(buf);
}


/**
 * Reshape array (without strides)
 *
 * Only flagged dims may flow
 */
void md_reshape(int D, unsigned long flags, const long odims[D], void* optr, const long idims[D], const void* iptr, size_t size)
{
	assert(md_calc_size(D, odims) == md_calc_size(D, idims));
	assert(md_check_equal_dims(D, odims, idims, ~flags));

	long ostrs[D];
	md_calc_strides(D, ostrs, odims, size);

	long istrs[D];
	md_calc_strides(D, istrs, idims, size);

	if (md_check_equal_dims(D, ostrs, istrs, ~flags)) {	// strides consistent!

		md_copy(D, odims, optr, iptr, size);

	} else {

		md_reshape2(D, flags, odims, ostrs, optr, idims, istrs, iptr, size);
	}
}



bool md_compare2(int D, const long dims[D], const long str1[D], const void* src1,
			const long str2[D], const void* src2, size_t size)
{
	__block bool eq = true;

	const long (*nstr[2])[D] = { (const long (*)[D])str1, (const long (*)[D])str2 };

	NESTED(void, nary_cmp, (struct nary_opt_data_s* opt_data, void* ptrs[]))
	{
		size_t size2 = (size_t)((long)size * opt_data->size);

		bool eq2 = (0 == memcmp(ptrs[0], ptrs[1], size2));
#pragma 	omp atomic
		eq &= eq2;
	};

	optimized_nop(2, 0u, D, dims, nstr, (void*[2]){ (void*)src1, (void*)src2 }, (size_t[2]){ size, size }, nary_cmp);

	if (is_mpi(src1) || is_mpi(src2))
		mpi_reduce_land(1, &eq);

	return eq;
}


bool md_compare(int D, const long dims[D], const void* src1, const void* src2, size_t size)
{
	long str[D];
	md_calc_strides(D, str, dims, size);

	return md_compare2(D, dims, str, src1, str, src2, size);
}





static void md_septrafo_r(int D, int R, long dimensions[D], unsigned long flags, const long strides[D], void* ptr, md_trafo_fun_t fun)
{
	if (0 == R--)
		return;

	md_septrafo_r(D, R, dimensions, flags, strides, ptr, fun);

        if (MD_IS_SET(flags, R)) {

                void* nptr[1] = { ptr };
                const long* nstrides[1] = { strides };

		long dimsR = dimensions[R];
		long strsR = strides[R]; // because of clang

                dimensions[R] = 1;      // we made a copy in md_septrafo2

		NESTED(void, nary_septrafo, (void* ptr[]))
		{
			fun(dimsR, strsR, ptr[0]);
		};

                //md_nary_parallel(1, D, dimensions, nstrides, nptr, &data, nary_septrafo);
                md_nary(1, D, dimensions, nstrides, nptr, nary_septrafo);
                dimensions[R] = dimsR;
        }
}

/**
 * Apply a separable transformation along selected dimensions.
 *
 */
void md_septrafo2(int D, const long dimensions[D], unsigned long flags, const long strides[D], void* ptr, md_trafo_fun_t fun)
{
        long dimcopy[D];
	md_copy_dims(D, dimcopy, dimensions);

        md_septrafo_r(D, D, dimcopy, flags, strides, ptr, fun);
}



/**
 * Apply a separable transformation along selected dimensions.
 *
 */
void md_septrafo(int D, const long dims[D], unsigned long flags, void* ptr, size_t size, md_trafo_fun_t fun)
{
        md_septrafo2(D, dims, flags, MD_STRIDES(D, dims, size), ptr, fun);
}



/**
 * Copy diagonals from array specified by flags (with strides)
 *
 * dst(i, i, :, i, :) = src(i, i, :, i, :)
 *
 */
void md_copy_diag2(int D, const long dims[D], unsigned long flags, const long str1[D], void* dst, const long str2[D], const void* src, size_t size)
{
	long stride1 = 0;
	long stride2 = 0;
	long count = -1;

	for (int i = 0; i < D; i++) {

		if (MD_IS_SET(flags, i)) {

			if (count < 0)
				count = dims[i];

			assert(dims[i] == count);

			stride1 += str1[i];
			stride2 += str2[i];
		}
	}

	long xdims[D];
	md_select_dims(D, ~flags, xdims, dims);

	for (long i = 0; i < count; i++)
		md_copy2(D, xdims, str1, dst + i * stride1, str2, src + i * stride2, size);
}



/**
 * Copy diagonals from array specified by flags (without strides)
 *
 * dst(i ,i ,: ,i , :) = src(i ,i ,: ,i ,:)
 *
 */
void md_copy_diag(int D, const long dims[D], unsigned long flags, void* dst, const void* src, size_t size)
{
	long str[D];
	md_calc_strides(D, str, dims, size);

	md_copy_diag2(D, dims, flags, str, dst, str, src, size);
}



/**
 * Fill diagonals specified by flags with value (without strides)
 *
 * dst(i, i, :, i, :) = src[0]
 *
 */
void md_fill_diag(int D, const long dims[D], unsigned long flags, void* dst, const void* src, size_t size)
{
	long str2[D];
	md_singleton_strides(D, str2);

	md_copy_diag2(D, dims, flags, MD_STRIDES(D, dims, size), dst, str2, src, size);
}



static void md_circ_shift_inpl2(int D, const long dims[D], const long center[D], const long strs[D], void* dst, size_t size)
{
#if 0
	long dims1[D];
	long dims2[D];

	md_copy_dims(D, dims1, dims);
	md_copy_dims(D, dims2, dims);

	int i;

	for (i = 0; i < D; i++) {
		if (0 != center[i]) {

			dims1[i] = center[i];
			dims2[i] = dims[i] - center[i];
			break;
		}
	}

	if (i == D)
		return;

	long off = strs[i] * center[i];

	// cool but slow, instead we want to have a chain of swaps

	md_flip2(D, dims, MD_BIT(i), strs, dst, strs, dst, size);
	md_flip2(D, dims1, MD_BIT(i), strs, dst, strs, dst, size);
	md_flip2(D, dims2, MD_BIT(i), strs, dst + off, strs, dst + off, size);

	// also not efficient, we want to merge the chain of swaps

	long center2[D];
	md_copy_dims(D, center2, center);
	center2[i] = 0;

	md_circ_shift_inpl2(D, dims, center2, strs, dst, size);
#else
	// use tmp for now
	int i;

	for (i = 0; i < D; i++)
		if (0 != center[i])
			break;

	if (i == D)
		return;

	long tmp_strs[D];
	md_calc_strides(D, tmp_strs, dims, size);

	void* tmp = md_alloc_sameplace(D, dims, size, dst);

	md_copy2(D, dims, tmp_strs, tmp, strs, dst, size);
	md_circ_shift2(D, dims, center, strs, dst, tmp_strs, tmp, size);

	md_free(tmp);
#endif
}

/**
 * Circularly shift array (with strides)
 *
 * dst[mod(i + center)] = src[i]
 *
 */
void md_circ_shift2(int D, const long dimensions[D], const long center[D], const long str1[D], void* dst, const long str2[D], const void* src, size_t size)
{
	long pos[D];

	for (int i = 0; i < D; i++) {	// FIXME: it would be better to calc modulo

		pos[i] = center[i];

		while (pos[i] < 0)
			pos[i] += dimensions[i];
	}

	int i = 0;		// FIXME :maybe we should search the other way?

	while ((i < D) && (0 == pos[i]))
		i++;

	if (D == i) {

		md_copy2(D, dimensions, str1, dst, str2, src, size);
		return;
	}

	if (dst == src) {

		assert(str1 == str2);

		md_circ_shift_inpl2(D, dimensions, pos, str1, dst, size);
		return;
	}

	long shift = pos[i];

	assert(shift != 0);

	long dim1[D];
	long dim2[D];

	md_copy_dims(D, dim1, dimensions);
	md_copy_dims(D, dim2, dimensions);

	dim1[i] = shift;
	dim2[i] = dimensions[i] - shift;

	assert((dim1[i] >= 0) && (dim2[i] >= 0));

	pos[i] = 0;

	//printf("%d: %ld %ld %d\n", i, dim1[i], dim2[i], sizeof(dimensions));
	md_circ_shift2(D, dim1, pos, str1, dst, str2, src + dim2[i] * str2[i], size);
	md_circ_shift2(D, dim2, pos, str1, dst + dim1[i] * str1[i], str2, src, size);
}




/**
 * Circularly shift array (without strides)
 *
 * dst[mod(i + center)] = src[i]
 *
 */
void md_circ_shift(int D, const long dimensions[D], const long center[D], void* dst, const void* src, size_t size)
{
	long strides[D];
	md_calc_strides(D, strides, dimensions, size);

	md_circ_shift2(D, dimensions, center, strides, dst, strides, src, size);
}



/**
 * Circularly extend array (with strides)
 *
 */
void md_circ_ext2(int D, const long dims1[D], const long strs1[D], void* dst, const long dims2[D], const long strs2[D], const void* src, size_t size)
{
	long ext[D];

	for (int i = 0; i < D; i++) {

		ext[i] = dims1[i] - dims2[i];

		assert(ext[i] >= 0);
		assert(ext[i] <= dims2[i]);
	}

	int i = 0;		// FIXME :maybe we should search the other way?
	while ((i < D) && (0 == ext[i]))
		i++;

	if (D == i) {

		md_copy2(D, dims1, strs1, dst, strs2, src, size);
		return;
	}

	long dims1_crop[D];
	long dims2_crop[D];
	long ext_dims[D];

	md_copy_dims(D, dims1_crop, dims1);
	md_copy_dims(D, dims2_crop, dims2);
	md_copy_dims(D, ext_dims, dims1);

	dims1_crop[i] = dims2[i];
	dims2_crop[i] = ext[i];
	ext_dims[i] = ext[i];

	ext[i] = 0;

	//printf("%d: %ld %ld %d\n", i, dim1[i], dim2[i], sizeof(dimensions));
	md_circ_ext2(D, dims1_crop, strs1, dst, dims2, strs2, src, size);
	md_circ_ext2(D, ext_dims, strs1, dst + dims2[i] * strs1[i], dims2_crop, strs2, src, size);
}




/**
 * Circularly extend array (without strides)
 *
 */
void md_circ_ext(int D, const long dims1[D],  void* dst, const long dims2[D], const void* src, size_t size)
{
	md_circ_ext2(D, dims1, MD_STRIDES(D, dims1, size), dst,
			dims2, MD_STRIDES(D, dims2, size), src, size);
}



/**
 * Periodically extend array (with strides)
 *
 */
void md_periodic2(int D, const long dims1[D], const long strs1[D], void* dst, const long dims2[D], const long strs2[D], const void* src, size_t size)
{
	long dims1B[2 * D];
	long strs1B[2 * D];
	long strs2B[2 * D];

	for (int i = 0; i < D; i++) {

		assert(0 == dims1[i] % dims2[i]);

		// blocks
		dims1B[2 * i + 0] = dims2[i];
		strs1B[2 * i + 0] = strs1[i];

		strs2B[2 * i + 0] = strs2[i];

		// periodic copies
		dims1B[2 * i + 0] = dims1[i] / dims2[i];
		strs1B[2 * i + 0] = strs1[i] * dims2[i];

		strs2B[2 * i + 0] = 0;
	}

	md_copy2(D, dims1B, strs1B, dst, strs2B, src, size);
}



/**
 * Periodically extend array (without strides)
 *
 */
void md_periodic(int D, const long dims1[D], void* dst, const long dims2[D], const void* src, size_t size)
{
	md_periodic2(D, dims1, MD_STRIDES(D, dims1, size), dst,
			dims2, MD_STRIDES(D, dims2, size), src, size);
}


void md_mask_compress(int D, const long dims[D], long M, uint32_t dst[static M], const float* src)
{
	long N = md_calc_size(D, dims);

	assert(M == (N + 31) / 32);

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		cuda_mask_compress(N, dst, src);
		return;
	}
#endif 

#pragma omp parallel for
	for (long i = 0; i < M; i++) {

		uint32_t result = 0;

		for (long j = 0; j < 32; j++) {

			if ((32 * i + j) >= N)
				continue;

			if (0. != src[(32 * i + j)])
				result = MD_SET(result, j);
		}

		dst[i] = result;
	}
}


void md_mask_decompress(int D, const long dims[D], float* dst, long M, const uint32_t src[static M])
{
	long N = md_calc_size(D, dims);

	assert(M == (N + 31) / 32);

#ifdef USE_CUDA
	if (cuda_ondevice(src)) {

		cuda_mask_decompress(N, dst, src);
		return;
	}
#endif 

#pragma omp parallel for
	for (long i = 0; i < M; i++) {

		for (long j = 0; j < 32; j++) {

			if ((32 * i + j) >= N)
				continue;

			dst[32 * i + j] = MD_IS_SET(src[i], j) ? 1. : 0.;
		}
	}
}


/**
 * Allocate CPU memory
 *
 * return pointer to CPU memory
 */
void* md_alloc(int D, const long dimensions[D], size_t size)
{
	return xmalloc((size_t)(md_calc_size(D, dimensions) * (long)size));
}



/**
 * Allocate CPU memory and clear
 *
 * return pointer to CPU memory
 */
void* md_calloc(int D, const long dimensions[D], size_t size)
{
	void* ptr = md_alloc(D, dimensions, size);

	md_clear(D, dimensions, ptr, size);

	return ptr;
}



#ifdef USE_CUDA
/**
 * Allocate GPU memory
 *
 * return pointer to GPU memory
 */
void* md_alloc_gpu(int D, const long dimensions[D], size_t size)
{
	return cuda_malloc(md_calc_size(D, dimensions) * (long)size);
}



/**
 * Allocate GPU memory and copy from CPU pointer
 *
 * return pointer to GPU memory
 */
void* md_gpu_move(int D, const long dims[D], const void* ptr, size_t size)
{
	if (NULL == ptr)
		return NULL;

	void* gpu_ptr = md_alloc_gpu(D, dims, size);

	md_copy(D, dims, gpu_ptr, ptr, size);

	return gpu_ptr;
}
#endif


/**
 * Allocate virtual distributed memory
 */
void* md_alloc_mpi(int D, unsigned long f, const long dimensions[D], size_t size)
{
	auto hint = hint_mpi_create(f, D, dimensions);
	void* ret = vptr_alloc(D, dimensions, size, hint);

	vptr_hint_free(hint);

	return ret;
}

/**
 * Allocate MPI memory and copy from pointer
 */
void* md_mpi_move(int D, unsigned long f, const long dims[D], const void* ptr, size_t size)
{
	if (NULL == ptr)
		return NULL;

	void* mpi_ptr = md_alloc_mpi(D, f, dims, size);

	md_copy(D, dims, mpi_ptr, ptr, size);

	return mpi_ptr;
}

/**
 * Allocate MPI memory and move from pointer
 */
void* md_mpi_moveF(int D, unsigned long f, const long dims[D], const void* ptr, size_t size)
{
	if (NULL == ptr)
		return NULL;

	auto hint = hint_mpi_create(f, D, dims);
	void* ret = vptr_wrap(D, dims, size, ptr, hint, true, false);

	vptr_hint_free(hint);

	return ret;
}

/**
 * Register usual memory as didtributed pointer.
 * If writeback, all data in mpi pointer is synced back to wrapped pointer on free.
 */
void* md_mpi_wrap(int D, unsigned long f, const long dims[D], const void* ptr, size_t size, bool writeback)
{
	if (NULL == ptr)
		return NULL;

	auto hint = hint_mpi_create(f, D, dims);
	void* ret = vptr_wrap(D, dims, size, ptr, hint, false, writeback);

	vptr_hint_free(hint);

	return ret;
}


/**
 * Allocate memory on the same device (CPU/GPU) place as ptr
 *
 * return pointer to CPU memory if ptr is in CPU or to GPU memory if ptr is in GPU
 */
void* md_alloc_sameplace(int D, const long dimensions[D], size_t size, const void* ptr)
{
	void* ret = vptr_alloc_sameplace(D, dimensions, size, ptr);

	if (NULL != ret)
		return ret;

#ifdef USE_CUDA
	return (cuda_ondevice(ptr) ? md_alloc_gpu : md_alloc)(D, dimensions, size);
#else
	assert(NULL != ptr);
	return md_alloc(D, dimensions, size);
#endif
}


/**
 * Free CPU/GPU memory
 *
 */
void md_free(const void* ptr)
{
	if (vptr_free(ptr))
		return;

#ifdef USE_CUDA
	if (cuda_ondevice(ptr))
		cuda_free((void*)ptr);
	else
#endif
	xfree(ptr);
}


int md_max_idx(unsigned long flags)
{
	int i = -1;

	for ( ; 0 != flags; i++)
		flags /= 2;

	return i;
}

int md_min_idx(unsigned long flags)
{
	return ffsl((long)flags) - 1;
}

/**
 * Convert flat index to pos
 *
 */
void md_unravel_index(int D, long pos[D], unsigned long flags, const long dims[D], long index)
{
	long ind = index;

	for (int d = 0; d < D; ++d) {

		if (!MD_IS_SET(flags, d))
			continue;

		pos[d] = ind % dims[d];
		ind /= dims[d];
	}
}

/**
 * Convert pos to flat index
 *
 */
long md_ravel_index(int D, const long pos[D], unsigned long flags, const long dims[D])
{
	long ind = 0;

	for (int d = D; d > 0; --d) {

		if (!MD_IS_SET(flags, d - 1))
			continue;
		
		ind *= dims[d - 1];
		ind += pos[d - 1];
	}

	return ind;
}


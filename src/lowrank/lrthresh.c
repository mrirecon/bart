/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Tao Zhang and Joseph Cheng.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2015 Frank Ong <frankong@berkeley.edu>
 * 2014 Tao Zhang
 * 2014 Joseph Cheng 
 * 2014 Jon Tamir 
 * 2014-2016 Martin Uecker
 */

#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <assert.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/blockproc.h"
#include "num/casorati.h"

#include "iter/thresh.h"

#include "lowrank/batchsvd.h"
#include "lowrank/svthresh.h"

#include "lrthresh.h"


struct lrthresh_data_s {

	INTERFACE(operator_data_t);

	float lambda;
	bool randshift;
	bool use_gpu;
	bool noise;
	int remove_mean; 

	long strs_lev[DIMS];
	long strs[DIMS];

	long dims_decom[DIMS];
	long dims[DIMS];

	unsigned long mflags;
	unsigned long flags;
	long levels;
	long blkdims[MAX_LEV][DIMS];
};

DEF_TYPEID(lrthresh_data_s);



static struct lrthresh_data_s* lrthresh_create_data(const long dims_decom[DIMS], bool randshift, unsigned long mflags, const long blkdims[MAX_LEV][DIMS], float lambda, bool noise, int remove_mean, bool use_gpu);
static void lrthresh_free_data(const operator_data_t* data);
static void lrthresh_apply(const operator_data_t* _data, float lambda, complex float* dst, const complex float* src);



/**
 * Intialize lrthresh operator
 *
 * @param dims_decom - decomposition dimensions
 * @param randshift - randshift boolean
 * @param mflags - selects which dimensions gets reshaped as the first dimension in matrix
 * @param blkdims - contains block dimensions for all levels
 * @param use_gpu - gpu boolean
 *
 */
const struct operator_p_s* lrthresh_create(const long dims_lev[DIMS], bool randshift, unsigned long mflags, const long blkdims[MAX_LEV][DIMS], float lambda, bool noise, int remove_mean, bool use_gpu)
{
	struct lrthresh_data_s* data = lrthresh_create_data(dims_lev, randshift, mflags, blkdims, lambda, noise, remove_mean, use_gpu);

	return operator_p_create(DIMS, dims_lev, DIMS, dims_lev, CAST_UP(data), lrthresh_apply, lrthresh_free_data);
}



/**
 * Intialize lrthresh data
 *
 * @param dims_decom - dimensions with levels at LEVEL_DIMS
 * @param randshift - randshift boolean
 * @param mflags - selects which dimensions gets reshaped as the first dimension in matrix
 * @param blkdims - contains block dimensions for all levels
 * @param use_gpu - gpu boolean
 *
 */
static struct lrthresh_data_s* lrthresh_create_data(const long dims_decom[DIMS], bool randshift, unsigned long mflags, const long blkdims[MAX_LEV][DIMS], float lambda, bool noise, int remove_mean, bool use_gpu)
{
	PTR_ALLOC(struct lrthresh_data_s, data);
	SET_TYPEID(lrthresh_data_s, data);

	data->randshift = randshift;
	data->mflags = mflags;
	data->lambda = lambda;
	data->noise = noise;
	data->remove_mean = remove_mean;

	// level dimensions
	md_copy_dims(DIMS, data->dims_decom, dims_decom);
	md_calc_strides(DIMS, data->strs_lev, dims_decom, CFL_SIZE);

	// image dimensions
	data->levels = dims_decom[LEVEL_DIM];
	md_select_dims(DIMS, ~LEVEL_FLAG, data->dims, dims_decom);
	md_calc_strides(DIMS, data->strs, data->dims, CFL_SIZE);

	// blkdims
	for(long l = 0; l < data->levels; l++) {

		for (long i = 0; i < DIMS; i++)
			data->blkdims[l][i] = blkdims[l][i];
	}

	data->use_gpu = use_gpu;
	
	return PTR_PASS(data);
}



/**
 * Free lrthresh operator
 */
static void lrthresh_free_data(const operator_data_t* _data)
{
	xfree(CAST_DOWN(lrthresh_data_s, _data));
}



/*
 * Return a random number between 0 and limit inclusive.
 */
static int rand_lim(int limit)
{
	int divisor = RAND_MAX / (limit + 1);
	int retval;

	do { 
		retval = rand() / divisor;

	} while (retval > limit);

	return retval;
}



/*
 * Low rank threhsolding for arbitrary block sizes
 */
static void lrthresh_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	struct lrthresh_data_s* data = CAST_DOWN(lrthresh_data_s, _data);

	float lambda = mu * data->lambda;

	long strs1[DIMS];
	md_calc_strides(DIMS, strs1, data->dims_decom, 1);

//#pragma omp parallel for
	for (int l = 0; l < data->levels; l++) {

		complex float* dstl = dst + l * strs1[LEVEL_DIM];
		const complex float* srcl = src + l * strs1[LEVEL_DIM];

		long blkdims[DIMS];
		long shifts[DIMS];
		long unshifts[DIMS];
		long zpad_dims[DIMS];
		long M = 1;

		for (unsigned int i = 0; i < DIMS; i++) {

			blkdims[i] = data->blkdims[l][i];
			zpad_dims[i] = (data->dims[i] + blkdims[i] - 1) / blkdims[i];
			zpad_dims[i] *= blkdims[i];

			if (MD_IS_SET(data->mflags, i))
				M *= blkdims[i];

			if (data->randshift)
				shifts[i] = rand_lim(MIN(blkdims[i] - 1, zpad_dims[i] - blkdims[i]));
			else
				shifts[i] = 0;

			unshifts[i] = -shifts[i];
		}

		long zpad_strs[DIMS];
		md_calc_strides(DIMS, zpad_strs, zpad_dims, CFL_SIZE);

		long blk_size = md_calc_size(DIMS, blkdims);
		long img_size = md_calc_size(DIMS, zpad_dims);
		long N = blk_size / M;
		long B = img_size / blk_size;

		if (data->noise && (l == data->levels - 1)) {

			M = img_size;
			N = 1;
			B = 1;
		}


		complex float* tmp;
#ifdef USE_CUDA
		tmp = (data->use_gpu ? md_alloc_gpu : md_alloc)(DIMS, zpad_dims, CFL_SIZE);
#else
		tmp = md_alloc(DIMS, zpad_dims, CFL_SIZE);
#endif

		md_circ_ext(DIMS, zpad_dims, tmp, data->dims, srcl, CFL_SIZE);

		md_circ_shift(DIMS, zpad_dims, shifts, tmp, tmp, CFL_SIZE);


		long mat_dims[2];
		basorati_dims(DIMS, mat_dims, blkdims, zpad_dims);

		complex float* tmp_mat;
#ifdef USE_CUDA
		tmp_mat = (data->use_gpu ? md_alloc_gpu : md_alloc)(2, mat_dims, CFL_SIZE);
#else
		tmp_mat = md_alloc(2, mat_dims, CFL_SIZE);
#endif
		// Reshape image into a blk_size x number of blocks matrix

		basorati_matrix(DIMS, blkdims, mat_dims, tmp_mat, zpad_dims, zpad_strs, tmp);

		batch_svthresh(M, N, mat_dims[1], lambda * GWIDTH(M, N, B), *(complex float (*)[mat_dims[1]][M][N])tmp_mat);

		//	for ( int b = 0; b < mat_dims[1]; b++ )
		//	svthresh(M, N, lambda * GWIDTH(M, N, B), tmp_mat, tmp_mat);

		basorati_matrixH(DIMS, blkdims, zpad_dims, zpad_strs, tmp, mat_dims, tmp_mat);

		md_circ_shift(DIMS, zpad_dims, unshifts, tmp, tmp, CFL_SIZE);

		md_resize(DIMS, data->dims, dstl, zpad_dims, tmp, CFL_SIZE);

		md_free(tmp);
		md_free(tmp_mat);
	}
}



/*
 * Nuclear norm calculation for arbitrary block sizes
 */
float lrnucnorm(const struct operator_p_s* op, const complex float* src)
{
	struct lrthresh_data_s* data = (struct lrthresh_data_s*)operator_p_get_data(op);

	long strs1[DIMS];
	md_calc_strides(DIMS, strs1, data->dims_decom, 1);
	float nnorm = 0.;


	for (int l = 0; l < data->levels; l++) {

		const complex float* srcl = src + l * strs1[LEVEL_DIM];

		long blkdims[DIMS];
		long blksize = 1;

		for (unsigned int i = 0; i < DIMS; i++) {

			blkdims[i] = data->blkdims[l][i];
			blksize *= blkdims[i];
		}

		if (1 == blksize) {

			for (long j = 0; j < md_calc_size(DIMS, data->dims); j++)
				nnorm += 2 * cabsf(srcl[j]);
				
			continue;
		}

		struct svthresh_blockproc_data* svdata = svthresh_blockproc_create(data->mflags, 0., 0);

		complex float* tmp;
#ifdef USE_CUDA
		tmp = (data->use_gpu ? md_alloc_gpu : md_alloc)(DIMS, data->dims, CFL_SIZE);
#else
		tmp = md_alloc(DIMS, data->dims, CFL_SIZE);
#endif

		//debug_print_dims(DP_DEBUG1, DIMS, data->dims);
		md_copy(DIMS, data->dims, tmp, srcl, CFL_SIZE);

		// Block SVD Threshold
		nnorm = blockproc(DIMS, data->dims, blkdims, (void*)svdata, nucnorm_blockproc, tmp, tmp);

		free(svdata);
		md_free(tmp);
	}

	return nnorm;
}




/*************
 * Block dimensions functions
 *************/


/**
 * Generates multiscale low rank block sizes
 *
 * @param blkdims - block sizes to be written
 * @param flags  - specifies which dimensions to do the blocks. The other dimensions will be the same as input
 * @param idims - input dimensions
 * @param blkskip - scale each level by blkskip to generate the next level
 *
 * returns number of levels
 */
long multilr_blkdims(long blkdims[MAX_LEV][DIMS], unsigned long flags, const long idims[DIMS], int blkskip, long initblk)
{
	// Multiscale low rank block sizes
	long tmp_block[DIMS];

	for (unsigned int i = 0; i < DIMS; i++) {

		if (MD_IS_SET(flags, i))
			tmp_block[i] = MIN(initblk, idims[i]);
		else
			tmp_block[i] = idims[i];
	}

	bool done;
	// Loop block_sizes
	long levels = 0;

	do {
		levels++;
		debug_printf(DP_INFO, "[\t");

		for (unsigned int i = 0; i < DIMS; i++) {

			blkdims[levels - 1][i] = tmp_block[i];
			debug_printf(DP_INFO, "%ld\t", blkdims[levels-1][i]);
		}

		debug_printf(DP_INFO, "]\n");


		done = true;

		for (unsigned int i = 0; i < DIMS; i++) {

			if (MD_IS_SET(flags, i) && (idims[i] != 1)) {

				tmp_block[i] = MIN(tmp_block[i] * blkskip, idims[i]);
				done = done && (blkdims[levels - 1][i] == idims[i]);
			}
		}
		
	} while(!done);

	return levels;
}



void add_lrnoiseblk(long* levels, long blkdims[MAX_LEV][DIMS], const long idims[DIMS])
{
	levels[0]++;
		
	debug_printf(DP_DEBUG1, "[\t");

	for (unsigned int i = 0; i < DIMS; i++) {

		blkdims[levels[0] - 1][i] = idims[i];

		debug_printf(DP_DEBUG1, "%ld\t", blkdims[levels[0] - 1][i]);
	}

	debug_printf(DP_DEBUG1, "]\n");
}



/**
 * Generates locally low rank block sizes
 *
 * @param blkdims - block sizes to be written
 * @param flags  - specifies which dimensions to do the blocks. The other dimensions will be the same as input
 * @param idims - input dimensions
 * @param llkblk - the block size
 *
 * returns number of levels = 1
 */
long llr_blkdims(long blkdims[MAX_LEV][DIMS], unsigned long flags, const long idims[DIMS], long llrblk)
{
	for (unsigned int i = 0; i < DIMS; i++) {

		if (MD_IS_SET(flags, i))
			blkdims[0][i] = MIN(llrblk, idims[i]);
		else
			blkdims[0][i] = idims[i];
	}

	return 1;
}



/**
 * Generates low rank + sparse block sizes
 *
 * @param blkdims - block sizes to be written
 * @param idims - input dimensions
 *
 * returns number of levels = 2
 */
long ls_blkdims(long blkdims[MAX_LEV][DIMS], const long idims[DIMS])
{
	for (unsigned int i = 0; i < DIMS; i++) {

		blkdims[0][i] = 1;
		blkdims[1][i] = idims[i];
	}

	return 2;
}


float get_lrthresh_lambda(const struct operator_p_s* o)
{
	const struct lrthresh_data_s* data = CAST_DOWN(lrthresh_data_s, operator_p_get_data(o));

	return data->lambda;
}

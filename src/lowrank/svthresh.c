/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 *
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
#include "num/blas.h"
#include "num/linalg.h"
#include "num/ops.h"
#include "num/iovec.h"
#include "num/fft.h"

#include "iter/thresh.h"

#include "svthresh.h"


float svthresh_nomeanu( long M, long N, float lambda, complex float* dst, const complex float* src)
{
	long MN = M*N;


	complex float* basis = md_alloc( 1, &M, CFL_SIZE );
	complex float* coeff = md_alloc( 1, &N, CFL_SIZE );
	complex float* tmp = md_alloc( 1, &MN, CFL_SIZE );

	for( int i = 0; i < M; i++)
		basis[i] = 1./sqrtf( M );


	md_clear( 1, &N, coeff, CFL_SIZE );
	md_clear( 1, &MN, tmp, CFL_SIZE );

	for( int j = 0; j < N; j++)
		for( int i = 0; i < M; i++)
			coeff[j] += basis[i] * src[i + j*M];

	for( int j = 0; j < N; j++)
		for( int i = 0; i < M; i++)
			tmp[i + j*M] = src[i + j*M] - coeff[j] * basis[i];

	svthresh(M, N, lambda , dst, tmp);

	for( int j = 0; j < N; j++)
		for( int i = 0; i < M; i++)
			dst[i + j*M] += coeff[j] * basis[i];


	return 0;

}


float svthresh_nomeanv( long M, long N, float lambda, complex float* dst, const complex float* src)
{
	long MN = M*N;


	complex float* basis = md_alloc( 1, &N, CFL_SIZE );
	complex float* coeff = md_alloc( 1, &M, CFL_SIZE );
	complex float* tmp = md_alloc( 1, &MN, CFL_SIZE );

	for( int i = 0; i < N; i++)
		basis[i] = 1. / sqrtf( N );

	md_clear( 1, &M, coeff, CFL_SIZE );
	md_clear( 1, &MN, tmp, CFL_SIZE );

	for( int j = 0; j < N; j++)
		for( int i = 0; i < M; i++)
			coeff[i] += basis[j] * src[i + j*M];
;

	for( int j = 0; j < N; j++)
		for( int i = 0; i < M; i++)
			tmp[i + j*M] = src[i + j*M] - coeff[i] * basis[j];


	svthresh(M, N, lambda , dst, tmp);

	for( int j = 0; j < N; j++)
		for( int i = 0; i < M; i++)
			dst[i + j*M] += coeff[i] * basis[j];


	return 0;

}

/**
 * Singular Value Thresholding
 *
 * @param M - matrix column size
 * @param N - matrix row size
 * @param lambda - regularization parameter
 * @param A - input/output matrix
 */
float svthresh(long M, long N, float lambda, complex float* dst, const complex float* src) //FIXME: destroys input
{


	long minMN = MIN(M,N);
	long dimsU[3] = {M,minMN,1};
	long dimsVT[3] = {minMN,N,1};
	long dimsS[3] = {minMN,1,1};
//	long dimsAA[3] = {minMN, minMN,1};

	long strsVT[3];
	long strsS[3];
	md_calc_strides(3, strsVT, dimsVT, CFL_SIZE);
	md_calc_strides(3, strsS, dimsS, FL_SIZE);


	complex float* U = md_alloc_sameplace(3, dimsU, CFL_SIZE, src);
	complex float* VT = md_alloc_sameplace(3, dimsVT, CFL_SIZE, src );
	float* S = md_alloc_sameplace(3, dimsS, FL_SIZE, src);

//	complex float* AA = md_alloc_sameplace(3, dimsAA, CFL_SIZE, src );
//	lapack_normal_multiply( M, N, (M > N), (complex float (*) [])AA, (const complex float (*) [])src );
	

	// SVD
	lapack_svd_econ(M, N, (complex float (*) []) U, (complex float (*) []) VT, S, (complex float (*) [N])src);

	// Thresh S
	md_softthresh(3, dimsS, lambda, 0, S, S);

	// VT = S * VT
	md_mul2( 3, dimsVT, strsVT, (float*) VT, strsVT, (float*) VT, strsS, S );
	md_mul2( 3, dimsVT, strsVT, ((float*) VT)+1, strsVT, ((float*) VT)+1, strsS, S );

	// dst = U * VT
	blas_matrix_multiply( M, N, minMN, (complex float (*) [N])dst, (const complex float (*) [minMN])U, (const complex float (*) [N])VT );

	md_free(U);
	md_free(VT);
	md_free(S);

	return 0;
}





float nuclearnorm(long M, long N, const complex float* d) { // FIXME: destroys input

	long minMN = MIN(M,N);
	long dimsU[3]	=	{M, minMN, 1};
	long dimsVT[3]	=	{minMN, N, 1};
	long dimsS[3]	=	{minMN, 1, 1};


	complex float* U = md_alloc_sameplace(3, dimsU, CFL_SIZE, d);
	complex float* VT = md_alloc_sameplace(3, dimsVT, CFL_SIZE, d );
	float* S = md_alloc_sameplace(3, dimsS, FL_SIZE, d);


	// SVD
	lapack_svd_econ(M, N, (complex float (*) []) U, (complex float (*) []) VT, S, (complex float (*) [N])d);

	float nnorm = 0.;
	for (int i = 0; i < minMN; i++)
		nnorm += S[i];

	md_free(U);
	md_free(VT);
	md_free(S);


	return nnorm;
}


float maxsingular(long M, long N, const complex float* d) {	// FIXME: destroys input
	long dimsU[2] = {M,N};
	long dimsV[2] = {N,N};
  
	complex float* U = md_alloc(2, dimsU, sizeof(complex float) );
	complex float* VT = md_alloc(2, dimsV, sizeof(complex float) );
	float* S = xmalloc( MIN(M,N) * sizeof(float) );

	// SVD
	lapack_svd_econ(M, N, (complex float (*) []) U, 
		 (complex float (*) []) VT, S, 
		 (complex float (*) [N])d);

	float value = S[0];

	md_free(U);
	md_free(VT);
	free(S);

	return value;
}



/***********
 * Blockproc functions
 ************/

struct svthresh_blockproc_data {
	unsigned long mflags;
	float lambda;
	int remove_mean;
};

struct svthresh_blockproc_data* svthresh_blockproc_create( unsigned long mflags, float lambda, int remove_mean )
{
	PTR_ALLOC(struct svthresh_blockproc_data, data);
	data->mflags = mflags;
	data->lambda = lambda;
	data->remove_mean = remove_mean;
	return data;
}

float svthresh_blockproc( const void* _data, const long blkdims[DIMS], complex float* dst, const complex float* src )
{
	const struct svthresh_blockproc_data* data = (const struct svthresh_blockproc_data*) _data;

	long M = 1;
	long N = md_calc_size( DIMS, blkdims );


	for ( unsigned int i = 0; i < DIMS; i++ )
	{
		if (MD_IS_SET(data->mflags, i))
		{
			M *= blkdims[i];
			N /= blkdims[i];
		}
	}

	if (data->remove_mean == 1)
		svthresh_nomeanu(M, N, data->lambda , dst, src);
	else if (data->remove_mean == 2)
		svthresh_nomeanv(M, N, data->lambda , dst, src);
	else if (data->remove_mean == 0)
		svthresh(M, N, data->lambda , dst, src);
	else
		assert(0);

	return 0;
	
}



float nucnorm_blockproc( const void* _data, const long blkdims[DIMS], complex float* dst, const complex float* src )
{
	UNUSED(dst);

	const struct svthresh_blockproc_data* data = (const struct svthresh_blockproc_data*) _data;

	long M = 1;
	long N = md_calc_size( DIMS, blkdims );


	for ( unsigned int i = 0; i < DIMS; i++ )
	{
		if (MD_IS_SET(data->mflags, i))
		{
			M *= blkdims[i];
			N /= blkdims[i];
		}
	}

	float G = sqrtf(M) + sqrtf(N);

	return G * nuclearnorm(M, N, src);
}



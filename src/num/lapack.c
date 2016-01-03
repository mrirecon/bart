/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2013. Tao Zhang.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2015 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2013 Tao Zhang <tao@mrsrl.stanford.edu>
 * 2014-2015 Frank Ong <frankong@berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * Wrapper functions for lapack.
 */
#include <math.h>
#include <complex.h>
#include <stdio.h>
#include <assert.h>

#ifdef USE_CUDA
#include "num/gpuops.h"

#include <cublas.h>
#endif


#include "misc/misc.h"

#include "num/lapack.h"


/* ATTENTION: blas and lapack use column-major matrices
 * while native C uses row-major. All matrices are
 * transposed to what one would expect.
 *
 * LAPACK svd destroys its input matrix
 **/


#ifdef USE_ACML
#if 1
// FIXME: check indices
extern void cheev(char jobz, char uplo, long N, complex float a[N][N], long lda, float w[N], long* info);
extern void zheev(char jobz, char uplo, long N, complex double a[N][N], long lda, double w[N], long* info);
extern void cgesdd(const char jobz, long M, long N, complex float A[M][N], long lda, float* S, complex float U[M][N], long ldu, complex float VH[M][N], long ldvt, const long* info);
extern void zgesdd(const char jobz, long M, long N, complex double A[M][N], long lda, double* S, complex double U[M][N], long ldu, complex double VH[M][N], long ldvt, const long* info);
extern void cgesvd(char jobu, char jobvt, long M, long N, complex float a[M][N], long lda, float* S, complex float u[M][N], long ldu, complex float vt[M][N], long ldvt, long *info);
extern void cgemm(const char transa, const char transb, long M, long N,  long K, const complex float* alpha, const complex float A[M][K], const long lda, const complex float B[K][N], const long ldb, const complex float* beta, complex float C[M][N], const long ldc );
extern void csyrk(char uplo, char transa, long N, long K, const complex float *alpha, const complex float A[K][N], const long lda, const complex float *beta, const complex float C[N][N], const long ldc);
extern void cpotrf_(char uplo, const long N, complex float A[N][N], long lda, long* info);
#else
// FIXME: this strategy would work but needs explicit casts below
#include <acml.h>
#undef complex
#define complex _Complex
#endif
#else
extern void cheev_(const char jobz[1], const char uplo[1], const long* N, complex float a[*N][*N], const long* lda, float w[*N], complex float* work, const long* lwork, float* rwork, long* info);
extern void zheev_(const char jobz[1], const char uplo[1], const long* N, complex double a[*N][*N], const long* lda, double w[*N], complex double* work, const long* lwork, double* rwork, long* info);
extern void cgesdd_(const char jobz[1], const long* M, const long* N, complex float A[*M][*N], const long lda[1], float* S, complex float U[*M][*N], const long* ldu, complex float VH[*M][*N], const long* ldvt, complex float* work, const long* lwork, float* rwork, const long* iwork, const long* info);
extern void zgesdd_(const char jobz[1], const long* M, const long* N, complex double A[*M][*N], const long lda[1], double* S, complex double U[*M][*N], const long* ldu, complex double VH[*M][*N], const long* ldvt, complex double* work, const long* lwork, double* rwork, const long* iwork, const long* info);
extern void cgesvd_(const char jobu[1], const char jobvt[1], const long* M, const long* N, complex float A[*M][*N], const long* lda, float* s, complex float U[*M][*N], long* ldu, complex float VH[*M][*N], long* ldvt, complex float* work, long* lwork, float* rwork, const long* iwork, long* info);
extern void cgemm_(const char transa[1], const char transb[1], const long* M, const long* N, const long* K, const complex float* alpha, const complex float A[*M][*K], const long* lda, const complex float B[*K][*N], const long* ldb, const complex float* beta, complex float C[*M][*N], const long* ldc );
extern void csyrk_(const char uplo[1], const char trans[1], const long* N, const long* K, const complex float* alpha, const complex float A[*N][*K], const long* lda, const complex float* beta, const complex float C[*N][*N], const long* ldc);
extern void cpotrf_(const char uplo[1], const long* N, complex float A[*N][*N], const long* lda, long* info);
#endif

void batch_svthresh(long M, long N, long num_blocks, float lambda, complex float* dst, const complex float* src)
{
	long info = 0;

	long minMN = MIN(M, N);
	// create u, v, s
	complex float* U = xmalloc(M * minMN * sizeof(complex float));
	complex float* VT = xmalloc(minMN * N * sizeof(complex float));
	float* S = xmalloc(minMN * sizeof(float));


#ifndef USE_ACML
	// create lrwork
	long lwork = -1;
	complex float work1[1];
	float* rwork = xmalloc(5 * N * sizeof(float));
	long* iwork = xmalloc(8 * minMN * sizeof(long));

	// get optimal block size, create work
	// i + j * lda
	cgesvd_("S", "S", &M, &N, (complex float (*)[N])src, &M, S, (complex float (*)[minMN])U, &M, (complex float (*)[N])VT, &minMN, work1, &lwork, rwork, iwork, &info);

	lwork = (int)work1[0];
	complex float* work = xmalloc(lwork * sizeof(complex float));
#endif


	// create AA
	complex float* AA = xmalloc(minMN * minMN * sizeof(complex float));


	for (int b = 0; b < num_blocks; b++) {

		const complex float* src_b = src + b * M * N;
		complex float* dst_b = dst + b * M * N;

		// Compute upper bound | A^T A |_inf
		float s_upperbound = 0;

		if (M <= N)
#ifdef USE_ACML
			csyrk('U', 'N', M, N, &(const complex float){ 1. }, (const complex float (*)[])src_b, M, &(const complex float){ 0. }, (const complex float (*)[])AA, minMN);
#else
			csyrk_("U", "N", &M, &N, &(const complex float){ 1. }, (const complex float (*)[])src_b, &M, &(const complex float){ 0. }, (const complex float (*)[])AA, &minMN);
#endif
		else
#ifdef USE_ACML
			csyrk('U', 'T', N, M, &(const complex float){ 1. }, (const complex float(*)[])src_b, M, &(const complex float){ 0. }, (const complex float (*)[])AA, minMN);
#else
			csyrk_("U", "T", &N, &M, &(const complex float){ 1. }, (const complex float(*)[])src_b, &M, &(const complex float){ 0. }, (const complex float (*)[]) AA, &minMN);
#endif



		// lambda_max( A ) <= max_i sum_j | a_i^T a_j |
		for (int i = 0; i < minMN; i++)
		{
			float s = 0;

			for (int j = 0; j < minMN; j++)
				s += cabsf(AA[MIN(i, j) + MAX(i, j) * minMN]);

			s_upperbound = MAX(s_upperbound, s);
		}

		if (s_upperbound < lambda * lambda) {

			for (int i = 0; i < M * N; i++)
				dst_b[i] = 0.;

		} else {


#ifdef USE_ACML
			cgesvd('S', 'S', M, N, (complex float (*)[])src_b, M, S, (complex float (*)[])U, M, (complex float (*)[])VT, minMN, &info);
#else
			cgesvd_("S", "S", &M, &N, (complex float (*)[])src_b, &M, S, (complex float (*)[])U, &M, (complex float (*)[]) VT, &minMN, work, &lwork, rwork, iwork, &info);
#endif


			// Soft Threshold
			for (int i = 0; i < minMN; i++ ) {

				float s = S[i] - lambda;

				s = (s + fabsf(s)) / 2.;

				for ( int j = 0; j < N; j++ )
					VT[i + j * minMN] *= s;
			}

#ifdef USE_ACML
			cgemm('N', 'N', M, N, minMN, &(complex float){ 1. }, (const complex float (*)[])U, M, (const complex float (*)[])VT, minMN, &(const complex float){ 0. }, (complex float (*)[])dst_b, M);
#else
			cgemm_("N", "N", &M, &N, &minMN, &(complex float){ 1. }, (const complex float (*)[])U, &M, (const complex float (*)[])VT, &minMN, &(complex float){ 0. }, (complex float (*)[])dst_b, &M);
#endif

		}
	}

	free(U);
	free(VT);
	free(S);
	free(AA);

#ifndef USE_ACML

	free(work);
	free(iwork);
	free(rwork);
#endif
}


void lapack_eig_double(long N, double eigenval[N], complex double matrix[N][N])
{
        long info = 0;


#ifdef USE_ACML
        zheev('V', 'U', N, matrix, N, eigenval, &info);
#else
	assert(N > 0);
	long lwork = -1;
	complex double work1[1];
	double* rwork = xmalloc((3 * N - 2) * sizeof(double));
        zheev_("V", "U", &N, matrix, &N, eigenval, work1, &lwork, rwork, &info);

	if (0 != info)
		goto err;

	lwork = (int)work1[0];
	complex double* work = xmalloc(lwork * sizeof(complex double));
	zheev_("V", "U", &N, matrix, &N, eigenval, work, &lwork, rwork, &info);
	free(work);
	free(rwork);
#endif
	if (0 != info) 
		goto err;

	return;

err:
        fprintf(stderr, "cheev failed\n");
	abort();
}


void lapack_eig(long N, float eigenval[N], complex float matrix[N][N])
{
        long info = 0;

#ifdef USE_ACML
        cheev('V', 'U', N, matrix, N, eigenval, &info);
#else
	assert(N > 0);
	long lwork = -1;
	complex float work1[1];
	float* rwork = xmalloc((3 * N - 2) * sizeof(float));
        cheev_("V", "U", &N, matrix, &N, eigenval, work1, &lwork, rwork, &info);

	if (0 != info)
		goto err;

	lwork = (int)work1[0];
	complex float* work = xmalloc(lwork * sizeof(complex float));
	cheev_("V", "U", &N, matrix, &N, eigenval, work, &lwork, rwork, &info);
	free(work);
	free(rwork);
#endif

	if (0 != info) 
		goto err;

	return;

err:
        fprintf(stderr, "cheev failed\n");
	abort();
}



void lapack_svd(long M, long N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M])
{
        long info = 0;
	//assert(M >= N);

#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		// add support for cuda 7 svd
		assert(0);

	} else
#endif 
	{
#ifdef USE_ACML
	cgesdd('A', M, N, A, M, S, U, M, VH, N, &info);
#else
	long lwork = -1;
	complex float work1[1];
	float* rwork = xmalloc(MIN(M, N) * MAX(5 * MIN(M, N) + 7, 2 * MAX(M, N) + 2 * MIN(M, N) + 1) * sizeof(float));
	long* iwork = xmalloc(8 * MIN(M, N) * sizeof(long));

	cgesdd_("A", &M, &N, A, &M, S, U, &M, VH, &N, work1, &lwork, rwork, iwork, &info);

	if (0 != info)
		goto err;

	lwork = (int)work1[0];
	complex float* work = xmalloc(MAX(1, lwork) * sizeof(complex float));
	cgesdd_("A", &M, &N, A, &M, S, U, &M, VH, &N, work, &lwork, rwork, iwork, &info);
	free(rwork);
	free(iwork);
	free(work);
#endif
	}
	if (0 != info) 
		goto err;

	return;

err:
	fprintf(stderr, "svd failed\n");
        abort();
}


void lapack_svd_econ(long M, long N,
	      complex float U[M][(N > M) ? M : N],
	      complex float VH[(N > M) ? M : N][N],
	      float S[(N > M) ? M : N], complex float A[M][N])
{
	long info = 0;

	long minMN = MIN(M, N);

#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		// add support for cuda 7 svd
		assert(0);

	} else
#endif 
	{

#ifdef USE_ACML
	cgesvd('S', 'S', M, N, A, M, S, U, M, VH, minMN, &info);
#else
	long lwork = -1;
	complex float work1[1];
	float* rwork = xmalloc(5 * N * sizeof(float));
	long* iwork = xmalloc(8 * minMN * sizeof(long));

	cgesvd_("S", "S", &M, &N, A, &M, S, U, &M, VH, &minMN, work1, &lwork, rwork, iwork, &info);

	if(0 != info)
		goto err;

	lwork = (int)work1[0];
	complex float* work = xmalloc(lwork * sizeof(complex float));
	cgesvd_("S", "S", &M, &N, A, &M, S, U, &M, VH, &minMN, work, &lwork, rwork, iwork, &info);

	free(work);
	free(iwork);
	free(rwork);
#endif
	}

	if(0 != info)
		goto err;
	return;

err:
	fprintf(stderr,"svd failed %ld\n", info);
	abort();
}


void lapack_svd_double(long M, long N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[M][N])
{
        long info = 0;
	//assert(M >= N);
	
#ifdef USE_ACML
	zgesdd('A', M, N, A, M, S, U, M, VH, N, &info);
#else
	long lwork = -1;
	complex double work1[1];
	double* rwork = xmalloc(MIN(M, N) * MAX(5 * MIN(M, N) + 7, 2 * MAX(M, N) + 2 * MIN(M, N) + 1) * sizeof(double));
	long* iwork = xmalloc(8 * MIN(M, N) * sizeof(long));

	zgesdd_("A", &M, &N, A, &M, S, U, &M, VH, &N, work1, &lwork, rwork, iwork, &info);

	if (0 != info)
		goto err;

	lwork = (int)work1[0];
	complex double* work = xmalloc(MAX(1, lwork) * sizeof(complex double));
	zgesdd_("A", &M, &N, A, &M, S, U, &M, VH, &N, work, &lwork, rwork, iwork, &info);

	free(rwork);
	free(iwork);
	free(work);
#endif

	if (0 != info) 
		goto err;

	return;

err:
	fprintf(stderr, "svd failed\n");
        abort();
}

#if 0
void matrix_multiply(long M, long N, long K, complex float C[M][N], complex float A[M][K], complex float B[K][N])
{
#ifdef USE_CUDA
	if (cuda_ondevice( A )) {

		cublasCgemm('N', 'N', M, N, K, make_cuFloatComplex(1.,0.), 
			    (const cuComplex *) A, M, 
			    (const cuComplex *) B, K, make_cuFloatComplex(0.,0.), 
			    (cuComplex *)  C, M);
	} else
#endif 
	{
#ifdef USE_ACML
		cgemm( 'N', 'N', M, N, K, &(complex float){1.} , A, M, B, K, &(complex float){0.}, C, M );
#else
		cgemm_("N", "N",  &M,  &N,  &K,  &(complex float){1.}, A, &M, B, &K,&(complex float){0.}, C, &M );
#endif
	}
}

#else

void lapack_matrix_multiply(long M, long N, long K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N])
{
	cgemm_sameplace('N', 'N', M, N, K, &(complex float){ 1. }, A, M, B, K, &(complex float){ 0. }, C, M);
}


void cgemm_sameplace(const char transa, const char transb, long M, long N,  long K, const complex float* alpha, const complex float A[M][K], const long lda, const complex float B[K][N], const long ldb, const complex float* beta, complex float C[M][N], const long ldc)
{
#ifdef USE_CUDA
	if (cuda_ondevice( A )) {

		cublasCgemm(transa, transb, M, N, K, *(cuComplex*)alpha,
				(const cuComplex*)A, lda,
				(const cuComplex*)B, ldb, *(cuComplex*)beta,
				(cuComplex*) C, ldc);
	} else
#endif 
	{
#ifdef USE_ACML
		cgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#else
		cgemm_(&transa, &transb, &M, &N, &K, alpha, A, &lda, B, &ldb, beta, C, &ldc);
#endif
	}
}
#endif


void lapack_cholesky(long N, complex float A[N][N])
{
	long info = 0;
#ifdef USE_ACML
	cpotrf('U', N, A, N, &info);
#else
	cpotrf_("U", &N, A, &N, &info);
#endif
}




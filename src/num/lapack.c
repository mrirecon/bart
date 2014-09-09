/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2013 Tao Zhang <tao@mrsrl.stanford.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * Wrapper functions for lapack.
 */

#include <complex.h>
#include <stdio.h>
#include <assert.h>

#ifdef USE_CUDA
#include "num/gpuops.h"

#ifdef USE_CULA
#include <cula_lapack_device.h>
#include <cula_types.h>
#include <cula_lapack.h>
#endif

#include <cublas.h>
#endif


#include "misc/misc.h"

#include "num/lapack.h"


#ifdef USE_ACML
#if 1
extern void cheev(char jobz, char uplo, long N, complex float a[N][N], long lda, float w[N], long* info);
extern void zheev(char jobz, char uplo, long N, complex double a[N][N], long lda, double w[N], long* info);
extern void cgesdd(const char jobz, long M, long N, complex float A[M][N], long lda, float* S, complex float U[M][N], long ldu, complex float VT[M][N], long ldvt, const long* info);
extern void zgesdd(const char jobz, long M, long N, complex double A[M][N], long lda, double* S, complex double U[M][N], long ldu, complex double VT[M][N], long ldvt, const long* info);
extern void cgesvd(char jobu, char jobvt, long M, long N, complex float a[M][N], long lda, float* S, complex float u[M][N], long ldu, complex float vt[M][N], long ldvt, long *info);
extern void cgemm(const char transa, const char transb, long M, long N,  long K, const complex float* alpha, const complex float A[M][K], const long lda, const complex float B[K][N], const long ldb, const complex float* beta, complex float C[M][N], const long ldc );
#else
// FIXME: this strategy would work but needs explicit casts below
#include <acml.h>
#undef complex
#define complex _Complex
#endif
#else
extern void cheev_(const char jobz[1], const char uplo[1], const long* N, complex float a[*N][*N], const long* lda, float w[*N], complex float* work, const long* lwork, float* rwork, long* info);
extern void zheev_(const char jobz[1], const char uplo[1], const long* N, complex double a[*N][*N], const long* lda, double w[*N], complex double* work, const long* lwork, double* rwork, long* info);
extern void cgesdd_(const char jobz[1], const long* M, const long* N, complex float A[*M][*N], const long lda[1], float* S, complex float U[*M][*N], const long* ldu, complex float VT[*M][*N], const long* ldvt, complex float* work, const long* lwork, float* rwork, const long* iwork, const long* info);
extern void zgesdd_(const char jobz[1], const long* M, const long* N, complex double A[*M][*N], const long lda[1], double* S, complex double U[*M][*N], const long* ldu, complex double VT[*M][*N], const long* ldvt, complex double* work, const long* lwork, double* rwork, const long* iwork, const long* info);
extern void cgesvd_(const char jobu[1], const char jobvt[1], const long* M, const long* N, complex float A[*M][*N], const long* lda, float* s, complex float U[*M][*N], long* ldu, complex float VT[*M][*N], long* ldvt, complex float* work, long* lwork, float* rwork, const long* iwork, long* info);
extern void cgemm_(const char transa[1], const char transb[1], const long* M, const long* N, const long* K, const complex float* alpha, const complex float A[*M][*K], const long* lda, const complex float B[*K][*N], const long* ldb, const complex float* beta, complex float C[*M][*N], const long* ldc );
#endif


void eigendecomp_double(long N, double eigenval[N], complex double matrix[N][N])
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


void eigendecomp(long N, float eigenval[N], complex float matrix[N][N])
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



void svd(long M, long N, complex float U[M][M], complex float VT[N][N], float S[(N > M) ? M : N], complex float A[M][N])
{
        long info = 0;
	//assert(M >= N);

#ifdef USE_CUDA
#ifdef USE_CULA
	if (cuda_ondevice( A )) {
		culaDeviceCgesvd( 'A', 'A', M, N, (culaDeviceFloatComplex *)A, M, (culaDeviceFloat *)S, (culaDeviceFloatComplex *)U, M, (culaDeviceFloatComplex *)VT, N );
	} else
#endif 
#endif 
	{
#ifdef USE_ACML
	cgesdd('A', M, N, A, M, S, U, M, VT, N, &info);
#else
	long lwork = -1;
	complex float work1[1];
	float* rwork = xmalloc(MIN(M, N) * MAX(5 * MIN(M, N) + 7, 2 * MAX(M, N) + 2 * MIN(M, N) + 1) * sizeof(float));
	long* iwork = xmalloc(8 * MIN(M, N) * sizeof(long));

	cgesdd_("A", &M, &N, A, &M, S, U, &M, VT, &N, work1, &lwork, rwork, iwork, &info);

	if (0 != info)
		goto err;

	lwork = (int)work1[0];
	complex float* work = xmalloc(MAX(1, lwork) * sizeof(complex float));
	cgesdd_("A", &M, &N, A, &M, S, U, &M, VT, &N, work, &lwork, rwork, iwork, &info);
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


void svd_econ(long M, long N, 
	      complex float U[M][(N > M) ? M : N], 
	      complex float VT[(N > M) ? M : N][N], 
	      float S[(N > M) ? M : N], complex float A[M][N])
{
	long info = 0;

	long minMN = MIN(M, N);

#ifdef USE_CUDA
#ifdef USE_CULA
	if (cuda_ondevice( A )) {
		culaDeviceCgesvd( 'S', 'S', M, N, (culaDeviceFloatComplex *)A, M, (culaDeviceFloat *)S, (culaDeviceFloatComplex *)U, M, (culaDeviceFloatComplex *)VT, minMN );
	} else
#endif 
#endif 

	{

#ifdef USE_ACML
	cgesvd('S', 'S', M, N, A, M, S, U, M, VT, minMN, &info);
#else
	long lwork = -1;
	complex float work1[1];
	float* rwork = xmalloc(5 * N * sizeof(float));
	long* iwork = xmalloc(8 * minMN * sizeof(long));

	cgesvd_("S", "S", &M, &N, A, &M, S, U, &M, VT, &minMN, work1, &lwork, rwork, iwork, &info);

	if(0 != info)
		goto err;

	lwork = (int)work1[0];
	complex float* work = xmalloc(lwork * sizeof(complex float));
	cgesvd_("S", "S", &M, &N, A, &M, S, U, &M, VT, &minMN, work, &lwork, rwork, iwork, &info);

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


void svd_double(long M, long N, complex double U[M][M], complex double VT[N][N], double S[(N > M) ? M : N], complex double A[M][N])
{
        long info = 0;
	//assert(M >= N);
	
#ifdef USE_ACML
	zgesdd('A', M, N, A, M, S, U, M, VT, N, &info);
#else
	long lwork = -1;
	complex double work1[1];
	double* rwork = xmalloc(MIN(M, N) * MAX(5 * MIN(M, N) + 7, 2 * MAX(M, N) + 2 * MIN(M, N) + 1) * sizeof(double));
	long* iwork = xmalloc(8 * MIN(M, N) * sizeof(long));

	zgesdd_("A", &M, &N, A, &M, S, U, &M, VT, &N, work1, &lwork, rwork, iwork, &info);

	if (0 != info)
		goto err;

	lwork = (int)work1[0];
	complex double* work = xmalloc(MAX(1, lwork) * sizeof(complex double));
	zgesdd_("A", &M, &N, A, &M, S, U, &M, VT, &N, work, &lwork, rwork, iwork, &info);
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

void matrix_multiply(long M, long N, long K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N])
{
	cgemm_sameplace('N', 'N', M, N, K, &(complex float){1.}, A, M, B, K, &(complex float){0.}, C, M);
}

void cgemm_sameplace(const char transa, const char transb, long M, long N,  long K, const complex float* alpha, const complex float A[M][K], const long lda, const complex float B[K][N], const long ldb, const complex float* beta, complex float C[M][N], const long ldc )
{
#ifdef USE_CUDA
	if (cuda_ondevice( A )) {
		cublasCgemm(transa, transb, M, N, K, *(cuComplex*)alpha,
				(const cuComplex *) A, lda, 
				(const cuComplex *) B, ldb, *(cuComplex*)beta, 
				(cuComplex *)  C, ldc);
	} else
#endif 

	{

#ifdef USE_ACML
		cgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc );
#else
		cgemm_(&transa, &transb,  &M,  &N,  &K,  alpha, A, &lda, B, &ldb, beta, C, &ldc );
#endif
	}
}
#endif

/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2019-2020 Moritz Blumenthal
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"

#ifdef USE_MACPORTS
#include <cblas_openblas.h>
#elif USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuComplex.h>
#include <cublas_v2.h>

#include "num/gpuops.h"
#endif

#include "blas.h"


#ifdef USE_CUDA
//blas2_* means, we use the new blas interface, i.e. scalar parameters are written an read by pointers.
//These pointers can point to cpu or gpu memory.

//blas_* uses the old interface where scalar parameters/results are provided/written by value/return.

static void cublas_error(int line, cublasStatus_t code)
{
	error("cublas error: %d in line %d \n", code, line);
}

#define CUBLAS_ERROR(x)	({ cublasStatus_t errval = (x); if (CUBLAS_STATUS_SUCCESS != errval) cublas_error(__LINE__, errval); })

static cublasHandle_t handle;
static _Bool handle_created = false;

static cublasHandle_t get_handle(void)
{
	if (!handle_created)
		CUBLAS_ERROR(cublasCreate(&handle));

	handle_created = true;

	return handle;
}

static void destroy_handle(void)
{
	CUBLAS_ERROR(cublasDestroy(handle));
	handle_created = false;
}

static void cublas_set_pointer_host(void)
{
	(void)get_handle();

	CUBLAS_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
}

static void cublas_set_pointer_device(void)
{
	(void)get_handle();

	CUBLAS_ERROR(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
}

static cublasOperation_t cublas_trans(char trans)
{
	if (('N' == trans) || ('n'==trans))
		return CUBLAS_OP_N;

	if (('T' == trans) || ('t'==trans))
		return CUBLAS_OP_T;

	if (('C' == trans) || ('c'==trans))
		return CUBLAS_OP_C;

	assert(0);
}
#endif




static void openblas_set_threads(void)
{
#ifndef USE_OPENBLAS
	return;
#else

#ifndef _OPENMP
	return;
#else
	if (1 != openblas_get_parallel())
		return; //pthread version of openblas

	#pragma omp critical
	openblas_set_num_threads(omp_in_parallel() ? 1 : omp_get_max_threads());
#endif
#endif
}




void blas2_cgemm(char transa, char transb, long M, long N, long K, const complex float* alpha, long lda, const complex float* A, long ldb, const complex float* B, const complex float* beta, long ldc, complex float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_device();

		cublasCgemm(get_handle(), cublas_trans(transa), cublas_trans(transb), M, N, K, (const cuComplex*)alpha,
			    (const cuComplex*)A, lda, (const cuComplex*)B, ldb, (const cuComplex*)beta, (cuComplex*)C, ldc);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, (void*)alpha, (void*)A, lda, (void*)B, ldb, (void*)beta, (void*)C, ldc);
}



void blas_cgemm(char transa, char transb, long M, long N,  long K, const complex float alpha, long lda, const complex float* A, long ldb, const complex float* B, const complex float beta, long ldc, complex float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		cublasCgemm(get_handle(), cublas_trans(transa), cublas_trans(transb), M, N, K, (const cuComplex*)(&alpha),
			    (const cuComplex*)A, lda, (const cuComplex*)B, ldb, (const cuComplex*)(&beta), (cuComplex*)C, ldc);

		return;
	}
#endif

	openblas_set_threads();

	cblas_cgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, (void*)(&alpha), (void*)A, lda, (void*)B, ldb, (void*)(&beta), (void*)C, ldc);
}



void blas2_cgemv(char trans, long M, long N, const complex float* alpha, long lda, const complex float* A, long incx, const complex float* x, complex float* beta, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_device();

		cublasCgemv(get_handle(), cublas_trans(trans), M, N, (const cuComplex*)alpha,
			    (const cuComplex*)A, lda, (const cuComplex*)x, incx, (const cuComplex*)beta, (cuComplex*)y, incy);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cgemv(CblasColMajor, ('T' == trans) ? CblasTrans : (('C' == trans) ? CblasConjTrans : CblasNoTrans),
		    M, N, (void*)alpha, (void*)A, lda, (void*)x, incx, (void*)beta, (void*)y, incy);
}



void blas_cgemv(char trans, long M, long N, complex float alpha, long lda, const complex float* A, long incx, const complex float* x, complex float beta, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		cublasCgemv(get_handle(), cublas_trans(trans), M, N, (const cuComplex*)&alpha,
			    (const cuComplex*)A, lda, (const cuComplex*)x, incx, (const cuComplex*)&beta, (cuComplex*)y, incy);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cgemv(CblasColMajor, ('T' == trans) ? CblasTrans : (('C' == trans) ? CblasConjTrans : CblasNoTrans),
		    M, N, (void*)&alpha, (void*)A, lda, (void*)x, incx, (void*)&beta, (void*)y, incy);
}



void blas2_cgeru(long M, long N, const complex float* alpha, long incx, const complex float* x, long incy, const complex float* y, long lda, complex float* A)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_device();

		cublasCgeru(get_handle(), M, N, (const cuComplex*)alpha,
			    (const cuComplex*)x, incx, (const cuComplex*)y, incy, (cuComplex*)A, lda);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cgeru(CblasColMajor, M, N, alpha, x, incx, y, incy, A, lda);
}



void blas_cgeru(long M, long N, complex float alpha, long incx, const complex float* x, long incy, const complex float* y, long lda, complex float* A)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		cublasCgeru(get_handle(), M, N, (const cuComplex*)&alpha,
			    (const cuComplex*)x, incx, (const cuComplex*)y, incy, (cuComplex*)A, lda);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cgeru(CblasColMajor, M, N, &alpha, x, incx, y, incy, (float*)A, lda);
}



void blas2_caxpy(long N, const complex float* alpha, long incx, const complex float* x, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_device();

		cublasCaxpy(get_handle(), N, (const cuComplex*)alpha, (const cuComplex*)x, incx, (cuComplex*)y, incy);

		return;
	}
#endif

	openblas_set_threads();

	cblas_caxpy(N, alpha, x, incx, y, incy);
}



void blas_caxpy(long N, const complex float alpha, long incx, const complex float* x, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_host();

		cublasCaxpy(get_handle(), N, (const cuComplex*)&alpha, (const cuComplex*)x, incx, (cuComplex*)y, incy);

		return;
	}
#endif

	openblas_set_threads();

	cblas_caxpy(N, &alpha, x, incx, y, incy);
}



void blas2_cscal(long N, const complex float* alpha, long incx, complex float* x)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_device();

		cublasCscal(get_handle(), N, (const cuComplex*)alpha, (cuComplex*)x, incx);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cscal(N, alpha, x, incx);
}



void blas_cscal(long N, const complex float alpha, long incx, complex float* x)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_host();

		cublasCscal(get_handle(), N, (const cuComplex*)&alpha, (cuComplex*)x, incx);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cscal(N, &alpha, x, incx);
}



void blas2_cdotu(complex float* result, long N, long incx, const complex float* x, long incy, const complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_device();

		cublasCdotu(get_handle(), N, (const cuComplex*)x, incx, (const cuComplex*)y, incy, (cuComplex*)result);

		return;
	}
#endif
	openblas_set_threads();

	cblas_cdotu_sub(N, x, incx, y, incy, (void*)result);
}



void blas2_sgemm(char transa, char transb, long M, long N, long K, const float* alpha, long lda, const float* A, long ldb, const float* B, const float* beta, long ldc, float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_device();

		cublasSgemm(get_handle(), cublas_trans(transa), cublas_trans(transb), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, *alpha, A, lda, B, ldb, *beta, C, ldc);
}



void blas_sgemm(char transa, char transb, long M, long N,  long K, const float alpha, long lda, const float* A, long ldb, const float* B, const float beta, long ldc, float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		cublasSgemm(get_handle(), cublas_trans(transa), cublas_trans(transb), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}



void blas2_sgemv(char trans, long M, long N, const float* alpha, long lda, const float* A, long incx, const float* x, float* beta, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_device();

		cublasSgemv(get_handle(), cublas_trans(trans), M, N, alpha,
			    (const float*)A, lda, x, incx, beta, y, incy);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sgemv(CblasColMajor, ('T' == trans) ? CblasTrans : CblasNoTrans, M, N, *alpha,
		    (const float*)A, lda, x, incx, *beta, y, incy);
}



void blas_sgemv(char trans, long M, long N, const float alpha, long lda, const float* A, long incx, const float* x, float beta, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		cublasSgemv(get_handle(), cublas_trans(trans), M, N, &alpha,
			    (const float*)A, lda, x, incx, &beta, y, incy);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sgemv(CblasColMajor, ('T' == trans) ? CblasTrans : CblasNoTrans, M, N, alpha,
		    (const float*)A, lda, x, incx, beta, y, incy);
}



void blas2_sger(long M, long N, const float* alpha, long incx, const float* x, long incy, const float* y, long lda, float* A)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_device();

		cublasSger(get_handle(), M, N, alpha, x, incx, y, incy, (float*)A, lda);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sger(CblasColMajor, M, N, *alpha, x, incx, y, incy, (float*)A, lda);
}



void blas_sger(long M, long N, const float alpha, long incx, const float* x, long incy, const float* y, long lda, float* A)
{
#ifdef USE_CUDA

	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		cublasSger(get_handle(), M, N, &alpha, x, incx, y, incy, (float*)A, lda);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sger(CblasColMajor, M, N, alpha, x, incx, y, incy, (float*)A, lda);
}



void blas2_saxpy(long N, const float* alpha, long incx, const float* x, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_device();

		cublasSaxpy(get_handle(), N, alpha, x, incx, y, incy);

		return;
	}
#endif
	openblas_set_threads();

	cblas_saxpy(N, *alpha, x, incx, y, incy);
}



void blas_saxpy(long N, const float alpha, long incx, const float* x, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_host();

		cublasSaxpy(get_handle(), N, &alpha, x, incx, y, incy);

		return;
	}
#endif
	openblas_set_threads();

	cblas_saxpy(N, alpha, x, incx, y, incy);
}



void blas2_sscal(long N, const float* alpha, long incx, float* x)
{
#ifdef USE_CUDA

	if (cuda_ondevice(x)) {

		cublas_set_pointer_device();

		cublasSscal(get_handle(), N, alpha, x, incx);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sscal(N, *alpha, x, incx);
}



void blas_sscal(long N, float alpha, long incx, float* x)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_host();

		cublasSscal(get_handle(), N, &alpha, x, incx);

		return;
	}
#endif
	openblas_set_threads();

	cblas_sscal(N, alpha, x, incx);
}



void blas2_sdot(float* result, long N, long incx, const float* x, long incy, const float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		cublas_set_pointer_device();

		cublasSdot(get_handle(), N, x, incx, y, incy, result);

		return;
	}
#endif
	openblas_set_threads();

	*result = cblas_sdot(N, x, incx, y, incy);
}



void blas_cdgmm(long M, long N, _Bool left_mul, const complex float* A, long lda, const complex float* x, long incx, complex float* C, long ldc)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublasCdgmm(get_handle(), left_mul ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			    M, N, (const cuComplex*)A, lda, (const cuComplex*)x, incx, (cuComplex*)C, ldc);

		return;
	}
#endif
	UNUSED(M);
	UNUSED(N);
	UNUSED(left_mul);
	UNUSED(A);
	UNUSED(lda);
	UNUSED(x);
	UNUSED(incx);
	UNUSED(C);
	UNUSED(ldc);

	assert(0);
}



void blas_sdgmm(long M, long N, _Bool left_mul, const float* A, long lda, const float* x, long incx, float* C, long ldc)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublasSdgmm(get_handle(), left_mul ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			    M, N, A, lda, x, incx, C, ldc);

		return;
	}
#endif
	UNUSED(M);
	UNUSED(N);
	UNUSED(left_mul);
	UNUSED(A);
	UNUSED(lda);
	UNUSED(x);
	UNUSED(incx);
	UNUSED(C);
	UNUSED(ldc);

	assert(0);
}



//B = alpha * op(A)
void blas_cmatcopy(char trans, long M, long N, complex float alpha, const complex float* A, long lda, complex float* B, long ldb)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		complex float zero = 0.;

		cublasCgeam(get_handle(), cublas_trans(trans), cublas_trans('N'),
			    M, N, (const cuComplex*)&alpha, (const cuComplex*)A, lda, (const cuComplex*)&zero, (const cuComplex*)B, ldb, (cuComplex*)B, ldb);

		return;
	}
#endif
	UNUSED(trans);
	UNUSED(M);
	UNUSED(N);
	UNUSED(alpha);
	UNUSED(lda);
	UNUSED(A);
	UNUSED(ldb);
	UNUSED(B);

	assert(0);
}



//B = alpha * op(A)
void blas2_cmatcopy(char trans, long M, long N, const complex float* alpha, const complex float* A, long lda, complex float* B, long ldb)
{
#ifdef USE_CUDA

	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		complex float* zero = cuda_malloc(8);
		cuda_clear(8, zero);

		cublasCgeam(get_handle(), cublas_trans(trans), cublas_trans('N'),
			    M, N, (const cuComplex*)alpha, (const cuComplex*)A, lda, (const cuComplex*)zero, (const cuComplex*)B, ldb, (cuComplex*)B, ldb);

		cuda_free(zero);

		return;
	}
#endif
	UNUSED(trans);
	UNUSED(M);
	UNUSED(N);
	UNUSED(alpha);
	UNUSED(lda);
	UNUSED(A);
	UNUSED(ldb);
	UNUSED(B);

	assert(0);
}



//B = alpha * op(A)
void blas_smatcopy(char trans, long M, long N, float alpha, const float* A, long lda, float* B, long ldb)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		float zero = 0.;

		cublasSgeam(get_handle(), cublas_trans(trans), cublas_trans('N'),
			    M, N, &alpha, A, lda, &zero, B, ldb, B, ldb);

		return;
	}
#endif
	UNUSED(trans);
	UNUSED(M);
	UNUSED(N);
	UNUSED(alpha);
	UNUSED(lda);
	UNUSED(A);
	UNUSED(ldb);
	UNUSED(B);

	assert(0);
}



//B = alpha * op(A)
void blas2_smatcopy(char trans, long M, long N, const float* alpha, const float* A, long lda, float* B, long ldb)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		cublas_set_pointer_host();

		float* zero = cuda_malloc(4);
		cuda_clear(4, zero);

		cublasSgeam(get_handle(), cublas_trans(trans), cublas_trans('N'),
			    M, N, alpha, A, lda, zero, B, ldb, B, ldb);

		cuda_free(zero);

		return;
	}
#endif
	UNUSED(trans);
	UNUSED(M);
	UNUSED(N);
	UNUSED(alpha);
	UNUSED(lda);
	UNUSED(A);
	UNUSED(ldb);
	UNUSED(B);

	assert(0);
}



void blas_csyrk(char uplo, char trans, long N, long K, const complex float alpha, long lda, const complex float A[][lda], complex float beta, long ldc, complex float C[][ldc])
{
	assert('U' == uplo);
	assert(('T' == trans) || ('N' == trans));

	cblas_csyrk(CblasColMajor, CblasUpper, ('T' == trans) ? CblasTrans : CblasNoTrans, N, K, (void*)&alpha, (void*)A, lda, (void*)&beta, (void*)C, ldc);
}



void blas_sger_fmac(long M, long N, float* A, const float* x, const float* y)
{
	blas_sger(M, N, 1., 1, x, 1, y, M, A);
}



void blas_gemv_zfmac(long M, long N, complex float* y, const complex float* A, char trans, const complex float* x)
{
	assert((trans == 'N') || (trans == 'T') || (trans == 'C'));

	blas_cgemv(trans,M, N, 1., M, A, 1, x, 1., 1, y);
}



void blas_gemv_fmac(long M, long N, float* y, const float* A, char trans, const float* x)
{
	assert((trans == 'N') || (trans == 'T'));

	blas_sgemv(trans,M, N, 1., M, A, 1, x, 1., 1, y);
}



void blas_matrix_multiply(long M, long N, long K, complex float C[N][M], const complex float A[K][M], const complex float B[N][K])
{
	blas_cgemm('N', 'N', M, N, K, 1. , M, (const complex float*)A, K, (const complex float*)B, 0., M, (complex float*)C);
}



void blas_matrix_zfmac(long M, long N, long K, complex float* C, const complex float* A, char transa, const complex float* B, char transb)
{
	assert((transa == 'N') || (transa == 'T') || (transa == 'C'));
	assert((transb == 'N') || (transb == 'T') || (transb == 'C'));

	long lda = (transa == 'N' ? M: K);
	long ldb = (transb == 'N' ? K: N);

	blas_cgemm(transa, transb, M, N, K, 1., lda, A, ldb, B, 1., M, C);
}

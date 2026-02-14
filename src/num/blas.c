/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016-2020. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir
 * 2016-2020 Martin Uecker
 * 2019-2021 Moritz Blumenthal
 */

#ifndef NO_BLAS

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

static void cublas_error(const char* file, int line, cublasStatus_t code)
{
	const char* err_str[] = {
#define ENTRY(x) [CUBLAS_STATUS_ ## x] = STRINGIFY(CONCAT(CUBLAS_STATUS_,x))
		ENTRY(SUCCESS), ENTRY(NOT_INITIALIZED), ENTRY(ALLOC_FAILED),
		ENTRY(INVALID_VALUE), ENTRY(ARCH_MISMATCH), ENTRY(MAPPING_ERROR),
		ENTRY(EXECUTION_FAILED), ENTRY(INTERNAL_ERROR), ENTRY(NOT_SUPPORTED),
		ENTRY(LICENSE_ERROR)
#undef ENTRY
	};

	error("cuBLAS Error: %s in %s:%d \n", err_str[code], file, line);
}

#define CUBLAS_ERROR(x)	({ cublasStatus_t errval = (x); if (CUBLAS_STATUS_SUCCESS != errval) cublas_error(__FILE__, __LINE__, errval); })
#define CUBLAS_CALL(x)		({ CUDA_ASYNC_ERROR_NOTE("before cuBLAS call"); cublas_set_gpulock(); cublasStatus_t errval = (x); cublas_unset_gpulock(); if (CUBLAS_STATUS_SUCCESS != errval) cublas_error(__FILE__, __LINE__, errval); CUDA_ASYNC_ERROR_NOTE("after cuBLAS call"); })

static cublasHandle_t handle_host[CUDA_MAX_STREAMS + 1];
static cublasHandle_t handle_device[CUDA_MAX_STREAMS + 1];

#ifdef _OPENMP
//FIXME: Following cuBLAS documentation, cuBLAS calls should be threadsafe
//	 However, tests/test-pics-multigpu fails with too many (16) threads
#include <omp.h>
static omp_lock_t gpulock[CUDA_MAX_STREAMS + 1];;
static void cublas_set_gpulock(void)
{
	omp_set_lock(&(gpulock[cuda_get_stream_id()]));
}

static void cublas_unset_gpulock(void)
{
	omp_unset_lock(&(gpulock[cuda_get_stream_id()]));
}
#else
static void cublas_set_gpulock(void)
{
	return;
}

static void cublas_unset_gpulock(void)
{
	return;
}
#endif

void cublas_init(void)
{
	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {

		CUBLAS_ERROR(cublasCreate(&(handle_host[i])));
		CUBLAS_ERROR(cublasCreate(&(handle_device[i])));

		CUBLAS_ERROR(cublasSetPointerMode(handle_host[i], CUBLAS_POINTER_MODE_HOST));
		CUBLAS_ERROR(cublasSetPointerMode(handle_device[i], CUBLAS_POINTER_MODE_DEVICE));

		CUBLAS_ERROR(cublasSetStream(handle_host[i], cuda_get_stream_by_id(i)));
		CUBLAS_ERROR(cublasSetStream(handle_device[i], cuda_get_stream_by_id(i)));

#ifdef _OPENMP
		omp_init_lock(&(gpulock[i]));
#endif
	}
}

void cublas_deinit(void)
{
	for (int i = 0; i < CUDA_MAX_STREAMS + 1; i++) {

		CUBLAS_ERROR(cublasDestroy(handle_device[i]));
		CUBLAS_ERROR(cublasDestroy(handle_host[i]));

#ifdef _OPENMP
		omp_destroy_lock(&(gpulock[i]));
#endif
	}
}



static cublasHandle_t get_handle_device(void)
{
	return handle_device[cuda_get_stream_id()];
}

static cublasHandle_t get_handle_host(void)
{
	return handle_host[cuda_get_stream_id()];
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



double cuda_asum(long size, const float* src)
{
	double result = 0;

	while (size > 0) {

		float tmp;
		CUBLAS_CALL(cublasSasum(get_handle_host(), MIN(size, INT_MAX / 4), src, 1, &tmp));

		result += tmp;

		src += INT_MAX / 4;
		size -= INT_MAX / 4;
	}

	return result;
}

void cuda_saxpy(long size, float* y, float alpha, const float* src)
{
//	printf("SAXPY %x %x %ld\n", y, src, size);

	while (size > 0) {

		CUBLAS_CALL(cublasSaxpy(get_handle_host(), MIN(size, INT_MAX / 4), &alpha, src, 1, y, 1));

		src += INT_MAX / 4;
		y += INT_MAX / 4;
		size -= INT_MAX / 4;
	}

}

void cuda_swap(long size, float* a, float* b)
{
	while (size > 0) {

		CUBLAS_CALL(cublasSswap(get_handle_host(), MIN(size, INT_MAX / 4), a, 1, b, 1));

		a += INT_MAX / 4;
		b += INT_MAX / 4;
		size -= INT_MAX / 4;
	}
}

#endif

#ifdef _OPENMP
#include <omp.h>

static bool cpulock_init = false;
static omp_nest_lock_t cpulock;

#ifdef USE_OPENBLAS
// <cblas.h> does not necessarily come from OpenBLAS
int openblas_get_parallel(void);
#endif

static bool use_lock(void)
{
#ifdef BLAS_THREADSAFE
	return false;
#endif

#ifdef USE_OPENBLAS
	//OpenMP is used for parallelization
	if (2 == openblas_get_parallel())
		return false;
#endif

	return true;
}

static void blas_cpu_set_lock(void)
{
	if (!cpulock_init) {

#pragma 	omp critical(cpulock_init)
		{
			if (!cpulock_init) {

				omp_init_nest_lock(&cpulock);
				cpulock_init = true;
			}
		}
	}

	if (use_lock())
		omp_set_nest_lock(&cpulock);
}

static void blas_cpu_unset_lock(void)
{
	if (use_lock())
		omp_unset_nest_lock(&cpulock);
}

#define BLAS_CALL(x)	({ blas_cpu_set_lock(); (x); blas_cpu_unset_lock(); })

#else

#define BLAS_CALL(x)	x

#endif




void blas2_cgemm(char transa, char transb, long M, long N, long K, const complex float* alpha, long lda, const complex float* A, long ldb, const complex float* B, const complex float* beta, long ldc, complex float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasCgemm(get_handle_device(), cublas_trans(transa), cublas_trans(transb), M, N, K, (const cuComplex*)alpha,
			    (const cuComplex*)A, lda, (const cuComplex*)B, ldb, (const cuComplex*)beta, (cuComplex*)C, ldc));

		return;
	}
#endif

	BLAS_CALL(cblas_cgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, (void*)alpha, (void*)A, lda, (void*)B, ldb, (void*)beta, (void*)C, ldc));
}



void blas_cgemm(char transa, char transb, long M, long N,  long K, const complex float alpha, long lda, const complex float* A, long ldb, const complex float* B, const complex float beta, long ldc, complex float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasCgemm(get_handle_host(), cublas_trans(transa), cublas_trans(transb), M, N, K, (const cuComplex*)(&alpha),
			    (const cuComplex*)A, lda, (const cuComplex*)B, ldb, (const cuComplex*)(&beta), (cuComplex*)C, ldc));

		return;
	}
#endif

	BLAS_CALL(cblas_cgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, (void*)(&alpha), (void*)A, lda, (void*)B, ldb, (void*)(&beta), (void*)C, ldc));
}



void blas2_cgemv(char trans, long M, long N, const complex float* alpha, long lda, const complex float* A, long incx, const complex float* x, complex float* beta, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasCgemv(get_handle_device(), cublas_trans(trans), M, N, (const cuComplex*)alpha,
			    (const cuComplex*)A, lda, (const cuComplex*)x, incx, (const cuComplex*)beta, (cuComplex*)y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_cgemv(CblasColMajor, ('T' == trans) ? CblasTrans : (('C' == trans) ? CblasConjTrans : CblasNoTrans),
		    M, N, (void*)alpha, (void*)A, lda, (void*)x, incx, (void*)beta, (void*)y, incy));
}



void blas_cgemv(char trans, long M, long N, complex float alpha, long lda, const complex float* A, long incx, const complex float* x, complex float beta, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasCgemv(get_handle_host(), cublas_trans(trans), M, N, (const cuComplex*)&alpha,
			    (const cuComplex*)A, lda, (const cuComplex*)x, incx, (const cuComplex*)&beta, (cuComplex*)y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_cgemv(CblasColMajor, ('T' == trans) ? CblasTrans : (('C' == trans) ? CblasConjTrans : CblasNoTrans),
		    M, N, (void*)&alpha, (void*)A, lda, (void*)x, incx, (void*)&beta, (void*)y, incy));
}



void blas2_cgeru(long M, long N, const complex float* alpha, long incx, const complex float* x, long incy, const complex float* y, long lda, complex float* A)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasCgeru(get_handle_device(), M, N, (const cuComplex*)alpha,
			    (const cuComplex*)x, incx, (const cuComplex*)y, incy, (cuComplex*)A, lda));

		return;
	}
#endif

	BLAS_CALL(cblas_cgeru(CblasColMajor, M, N, alpha, x, incx, y, incy, A, lda));
}



void blas_cgeru(long M, long N, complex float alpha, long incx, const complex float* x, long incy, const complex float* y, long lda, complex float* A)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasCgeru(get_handle_host(), M, N, (const cuComplex*)&alpha,
			    (const cuComplex*)x, incx, (const cuComplex*)y, incy, (cuComplex*)A, lda));

		return;
	}
#endif

	BLAS_CALL(cblas_cgeru(CblasColMajor, M, N, &alpha, x, incx, y, incy, (float*)A, lda));
}



void blas2_caxpy(long N, const complex float* alpha, long incx, const complex float* x, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasCaxpy(get_handle_device(), N, (const cuComplex*)alpha, (const cuComplex*)x, incx, (cuComplex*)y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_caxpy(N, alpha, x, incx, y, incy));
}



void blas_caxpy(long N, const complex float alpha, long incx, const complex float* x, long incy, complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasCaxpy(get_handle_host(), N, (const cuComplex*)&alpha, (const cuComplex*)x, incx, (cuComplex*)y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_caxpy(N, &alpha, x, incx, y, incy));
}



void blas2_cscal(long N, const complex float* alpha, long incx, complex float* x)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasCscal(get_handle_device(), N, (const cuComplex*)alpha, (cuComplex*)x, incx));

		return;
	}
#endif

	BLAS_CALL(cblas_cscal(N, alpha, x, incx));
}



void blas_cscal(long N, const complex float alpha, long incx, complex float* x)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasCscal(get_handle_host(), N, (const cuComplex*)&alpha, (cuComplex*)x, incx));

		return;
	}
#endif

	BLAS_CALL(cblas_cscal(N, &alpha, x, incx));
}



void blas2_cdotu(complex float* result, long N, long incx, const complex float* x, long incy, const complex float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasCdotu(get_handle_device(), N, (const cuComplex*)x, incx, (const cuComplex*)y, incy, (cuComplex*)result));

		return;
	}
#endif

	BLAS_CALL(cblas_cdotu_sub(N, x, incx, y, incy, (void*)result));
}



void blas2_sgemm(char transa, char transb, long M, long N, long K, const float* alpha, long lda, const float* A, long ldb, const float* B, const float* beta, long ldc, float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasSgemm(get_handle_device(), cublas_trans(transa), cublas_trans(transb), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));

		return;
	}
#endif

	BLAS_CALL(cblas_sgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, *alpha, A, lda, B, ldb, *beta, C, ldc));
}



void blas_sgemm(char transa, char transb, long M, long N,  long K, const float alpha, long lda, const float* A, long ldb, const float* B, const float beta, long ldc, float* C)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasSgemm(get_handle_host(), cublas_trans(transa), cublas_trans(transb), M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc));

		return;
	}
#endif

	BLAS_CALL(cblas_sgemm(CblasColMajor, ('T' == transa) ? CblasTrans : (('C' == transa) ? CblasConjTrans : CblasNoTrans),
		    ('T' == transb) ? CblasTrans : (('C' == transb) ? CblasConjTrans : CblasNoTrans),
		    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc));
}



void blas2_sgemv(char trans, long M, long N, const float* alpha, long lda, const float* A, long incx, const float* x, float* beta, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasSgemv(get_handle_device(), cublas_trans(trans), M, N, alpha,
			    A, lda, x, incx, beta, y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_sgemv(CblasColMajor, ('T' == trans) ? CblasTrans : CblasNoTrans, M, N, *alpha,
			A, lda, x, incx, *beta, y, incy));
}



void blas_sgemv(char trans, long M, long N, const float alpha, long lda, const float* A, long incx, const float* x, float beta, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasSgemv(get_handle_host(), cublas_trans(trans), M, N, &alpha,
			A, lda, x, incx, &beta, y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_sgemv(CblasColMajor, ('T' == trans) ? CblasTrans : CblasNoTrans, M, N, alpha,
			A, lda, x, incx, beta, y, incy));
}



void blas2_sger(long M, long N, const float* alpha, long incx, const float* x, long incy, const float* y, long lda, float* A)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasSger(get_handle_device(), M, N, alpha, x, incx, y, incy, A, lda));
		return;
	}
#endif

	BLAS_CALL(cblas_sger(CblasColMajor, M, N, *alpha, x, incx, y, incy, A, lda));
}



void blas_sger(long M, long N, const float alpha, long incx, const float* x, long incy, const float* y, long lda, float* A)
{
#ifdef USE_CUDA

	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasSger(get_handle_host(), M, N, &alpha, x, incx, y, incy, A, lda));
		return;
	}
#endif

	BLAS_CALL(cblas_sger(CblasColMajor, M, N, alpha, x, incx, y, incy, A, lda));
}



void blas2_saxpy(long N, const float* alpha, long incx, const float* x, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasSaxpy(get_handle_device(), N, alpha, x, incx, y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_saxpy(N, *alpha, x, incx, y, incy));
}



void blas_saxpy(long N, const float alpha, long incx, const float* x, long incy, float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasSaxpy(get_handle_host(), N, &alpha, x, incx, y, incy));

		return;
	}
#endif

	BLAS_CALL(cblas_saxpy(N, alpha, x, incx, y, incy));
}



void blas2_sscal(long N, const float* alpha, long incx, float* x)
{
#ifdef USE_CUDA

	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasSscal(get_handle_device(), N, alpha, x, incx));

		return;
	}
#endif

	BLAS_CALL(cblas_sscal(N, *alpha, x, incx));
}



void blas_sscal(long N, float alpha, long incx, float* x)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasSscal(get_handle_host(), N, &alpha, x, incx));

		return;
	}
#endif

	BLAS_CALL(cblas_sscal(N, alpha, x, incx));
}



void blas2_sdot(float* result, long N, long incx, const float* x, long incy, const float* y)
{
#ifdef USE_CUDA
	if (cuda_ondevice(x)) {

		CUBLAS_CALL(cublasSdot(get_handle_device(), N, x, incx, y, incy, result));

		return;
	}
#endif

	BLAS_CALL(*result = cblas_sdot(N, x, incx, y, incy));
}



void blas_cdgmm(long M, long N, bool left_mul, const complex float* A, long lda, const complex float* x, long incx, complex float* C, long ldc)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasCdgmm(get_handle_device(), left_mul ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			    M, N, (const cuComplex*)A, lda, (const cuComplex*)x, incx, (cuComplex*)C, ldc));

		return;
	}
#else
	(void)M; (void)N; (void)left_mul; (void)A; (void)lda; (void)x; (void)incx; (void)C; (void)ldc;
#endif
	assert(0);
}



void blas_sdgmm(long M, long N, bool left_mul, const float* A, long lda, const float* x, long incx, float* C, long ldc)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		CUBLAS_CALL(cublasSdgmm(get_handle_device(), left_mul ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT,
			    M, N, A, lda, x, incx, C, ldc));

		return;
	}
#else
	(void)M; (void)N; (void)left_mul; (void)A; (void)lda; (void)x; (void)incx; (void)C; (void)ldc;
#endif
	assert(0);
}



//B = alpha * op(A)
void blas_cmatcopy(char trans, long M, long N, complex float alpha, const complex float* A, long lda, complex float* B, long ldb)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		complex float zero = 0.;

		CUBLAS_CALL(cublasCgeam(get_handle_host(), cublas_trans(trans), cublas_trans('N'),
			    M, N, (const cuComplex*)&alpha, (const cuComplex*)A, lda, (const cuComplex*)&zero, (const cuComplex*)B, ldb, (cuComplex*)B, ldb));

		return;
	}
#else
	(void)trans; (void)M; (void)N; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
#endif
	assert(0);
}



//B = alpha * op(A)
void blas2_cmatcopy(char trans, long M, long N, const complex float* alpha, const complex float* A, long lda, complex float* B, long ldb)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		complex float* zero = cuda_malloc(8);
		cuda_clear(8, zero);

		CUBLAS_CALL(cublasCgeam(get_handle_host(), cublas_trans(trans), cublas_trans('N'),
			    M, N, (const cuComplex*)alpha, (const cuComplex*)A, lda, (const cuComplex*)zero, (const cuComplex*)B, ldb, (cuComplex*)B, ldb));

		cuda_free(zero);

		return;
	}
#else
	(void)trans; (void)M; (void)N; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
#endif
	assert(0);
}



//B = alpha * op(A)
void blas_smatcopy(char trans, long M, long N, float alpha, const float* A, long lda, float* B, long ldb)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		float zero = 0.;

		CUBLAS_CALL(cublasSgeam(get_handle_host(), cublas_trans(trans), cublas_trans('N'),
			    M, N, &alpha, A, lda, &zero, B, ldb, B, ldb));

		return;
	}
#else
	(void)trans; (void)M; (void)N; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
#endif
	assert(0);
}



//B = alpha * op(A)
void blas2_smatcopy(char trans, long M, long N, const float* alpha, const float* A, long lda, float* B, long ldb)
{
#ifdef USE_CUDA
	if (cuda_ondevice(A)) {

		float* zero = cuda_malloc(4);
		cuda_clear(4, zero);

		CUBLAS_CALL(cublasSgeam(get_handle_host(), cublas_trans(trans), cublas_trans('N'),
			    M, N, alpha, A, lda, zero, B, ldb, B, ldb));

		cuda_free(zero);

		return;
	}
#else
	(void)trans; (void)M; (void)N; (void)alpha; (void)A; (void)lda; (void)B; (void)ldb;
#endif
	assert(0);
}



void blas_csyrk(char uplo, char trans, long N, long K, const complex float alpha, long lda, const complex float A[][lda], complex float beta, long ldc, complex float C[][ldc])
{
	assert('U' == uplo);
	assert(('T' == trans) || ('N' == trans));

	cblas_csyrk(CblasColMajor, CblasUpper, ('T' == trans) ? CblasTrans : CblasNoTrans, N, K, &alpha, (void*)A, lda, (void*)&beta, (void*)C, ldc);
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

#endif


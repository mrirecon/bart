/* Copyright 2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <complex.h>

#include "misc/misc.h"

#ifdef USE_MACPORTS
#include <cblas_openblas.h>
#else
#include <cblas.h>
#endif

#ifdef USE_CUDA
#include "num/gpuops.h"

#include <cublas.h>
#endif

#include "blas.h"



void blas_cgemm(char transa, char transb, long M, long N,  long K, const complex float alpha, long lda, const complex float A[K][lda], long ldb, const complex float B[N][ldb], const complex float beta, long ldc, complex float C[N][ldc])
{
#ifdef USE_CUDA
#define CUCOMPLEX(x) (((union { cuComplex cu; complex float std; }){ .std = (x) }).cu)
        if (cuda_ondevice(A)) {

                cublasCgemm(transa, transb, M, N, K,  CUCOMPLEX(alpha),
                                (const cuComplex*)A, lda,
                                (const cuComplex*)B, ldb, CUCOMPLEX(beta),
                                (cuComplex*)C, ldc);
        } else
#endif
        cblas_cgemm(CblasColMajor, transa, transb, M, N, K, (void*)&alpha, (void*)A, lda, (void*)B, ldb, (void*)&beta, (void*)C, ldc);
}


void (blas_matrix_multiply)(long M, long N, long K, complex float C[N][M], const complex float A[K][M], const complex float B[N][K])
{
	blas_cgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1., M, A, K, B, 0., M, C);
}



void (blas_csyrk)(char uplo, char trans, long N, long K, const complex float alpha, long lda, const complex float A[][lda], complex float beta, long ldc, complex float C[][ldc])
{
	assert('U' == uplo);
	assert(('T' == trans) || ('N' == trans));

	cblas_csyrk(CblasColMajor, CblasUpper, ('T' == trans) ? CblasTrans : CblasNoTrans, N, K, (void*)&alpha, (void*)A, lda, (void*)&beta, (void*)C, ldc);
}




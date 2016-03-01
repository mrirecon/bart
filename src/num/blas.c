
#include <assert.h>
#include <complex.h>

#include "misc/misc.h"

#include <cblas.h>

#ifdef USE_CUDA
#include "num/gpuops.h"

#include <cublas.h>
#endif

#include "blas.h"


void blas_cgemm(char transa, char transb, long M, long N,  long K, const complex float alpha, long lda, const complex float A[K][lda], long ldb, const complex float B[N][ldb], const complex float beta, long ldc, complex float C[N][ldc])
{
#ifdef USE_CUDA
        if (cuda_ondevice(A)) {

                cublasCgemm(transa, transb, M, N, K, *(cuComplex*)&alpha,
                                (const cuComplex*)A, lda,
                                (const cuComplex*)B, ldb, *(cuComplex*)&beta,
                                (cuComplex*)C, ldc);
        } else
#endif
        cblas_cgemm(CblasColMajor, transa, transb, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}


void (blas_matrix_multiply)(long M, long N, long K, complex float C[N][M], const complex float A[K][M], const complex float B[N][K])
{
	blas_cgemm(CblasNoTrans, CblasNoTrans, M, N, K, 1., M, A, K, B, 0., M, C);
}



void (blas_csyrk)(char uplo, char trans, long N, long K, const complex float alpha, long lda, const complex float A[][lda], complex float beta, long ldc, complex float C[][ldc])
{
	assert('U' == uplo);
	assert(('T' == trans) || ('N' == trans));

	cblas_csyrk(CblasColMajor, CblasUpper, ('T' == trans) ? CblasTrans : CblasNoTrans, N, K, &alpha, A, lda, &beta, C, ldc);
}




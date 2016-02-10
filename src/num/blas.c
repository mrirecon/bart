
#include <complex.h>

#include "misc/misc.h"

#include <cblas.h>

#ifdef USE_CUDA
#include "num/gpuops.h"

#include <cublas.h>
#endif

#include "blas.h"


void blas_cgemm(char transa, char transb, long M, long N,  long K, const complex float alpha, const complex float A[M][K], const long lda, const complex float B[K][N], const long ldb, const complex float beta, complex float C[M][N], const long ldc)
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


void blas_matrix_multiply(long M, long N, long K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N])
{
	blas_cgemm('N', 'N', M, N, K, 1., A, M, B, K, 0., C, M);
}




#include <complex.h>

#include "misc/misc.h"

extern void blas_matrix_multiply(long M, long N, long K, complex float C[N][M], const complex float A[K][M], const complex float B[N][K]);

extern void blas_cgemm(char transa, char transb, long M, long N, long K, const complex float alpha, long lda, const complex float A[M][lda], long ldb, const complex float B[K][ldb], const complex float beta, long ldc, complex float C[M][ldc]);
extern void blas_csyrk(char uplow, char trans, long N, long K, complex float alpha, long lda, const complex float A[*][lda], complex float beta, long ldc, complex float C[*][ldc]);



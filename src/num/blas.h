
#include <complex.h>

extern void blas_matrix_multiply(long M, long N, long K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N]);
extern void blas_cgemm(char transa, char transb, long M, long N,  long K, const complex float alpha, const complex float A[M][K], const long lda, const complex float B[K][N], const long ldb, const complex float beta, complex float C[M][N], const long ldc);


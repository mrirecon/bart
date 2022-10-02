
#include <complex.h>

#include "misc/misc.h"

#ifdef USE_CUDA
void cublas_init(void);
void cublas_deinit(void);

extern double cuda_sdot(long size, const float* src1, const float* src2);
extern double cuda_norm(long size, const float* src1);
extern double cuda_asum(long size, const float* src);
extern void cuda_saxpy(long size, float* y, float alpha, const float* src);
extern void cuda_swap(long size, float* a, float* b);
#endif



extern void blas_cgemm(char transa, char transb, long M, long N, long K, const _Complex float alpha, long lda, const _Complex float* A, long ldb, const _Complex float* B, const _Complex float beta, long ldc, _Complex float* C);
extern void blas2_cgemm(char transa, char transb, long M, long N, long K, const _Complex float* alpha, long lda, const _Complex float* A, long ldb, const _Complex float* B, const _Complex float* beta, long ldc, _Complex float* C);
extern void blas_sgemm(char transa, char transb, long M, long N, long K, const float alpha, long lda, const  float* A, long ldb, const  float* B, const  float beta, long ldc,  float* C);
extern void blas2_sgemm(char transa, char transb, long M, long N, long K, const float* alpha, long lda, const  float* A, long ldb, const  float* B, const  float* beta, long ldc,  float* C);

extern void blas_cgemv(char trans, long M, long N, _Complex float alpha, long lda, const _Complex float* A, long incx, const _Complex float* x, _Complex float beta, long incy, _Complex float* y);
extern void blas2_cgemv(char trans, long M, long N, const _Complex float* alpha, long lda, const _Complex float* A, long incx, const _Complex float* x, _Complex float* beta, long incy, _Complex float* y);
extern void blas_sgemv(char trans, long M, long N, float alpha, long lda, const float* A, long incx, const float* x, float beta, long incy, float* y);
extern void blas2_sgemv(char trans, long M, long N, const float* alpha, long lda, const float* A, long incx, const float* x, float* beta, long incy, float* y);

extern void blas_sger(long M, long N, float alpha, long incx, const float* x, long incy, const float* y, long lda, float* A);
extern void blas2_sger(long M, long N, const float* alpha, long incx, const float* x, long incy, const float* y, long lda, float* A);
extern void blas_cgeru(long M, long N, _Complex float alpha, long incx, const _Complex float* x, long incy, const _Complex float* y, long lda, _Complex float* A);
extern void blas2_cgeru(long M, long N, const _Complex float* alpha, long incx, const _Complex float* x, long incy, const _Complex float* y, long lda, _Complex float* A);

extern void blas2_caxpy(long N, const _Complex float* alpha, long incx, const _Complex float* x, long incy, _Complex float* y);
extern void blas_caxpy(long N, _Complex float alpha, long incx, const _Complex float* x, long incy, _Complex float* y);
extern void blas2_saxpy(long N, const float* alpha, long incx, const float* x, long incy, float* y);
extern void blas_saxpy(long N, float alpha, long incx, const float* x, long incy, float* y);

extern void blas2_cscal(long N, const _Complex float* alpha, long incx, _Complex float* x);
extern void blas_cscal(long N, _Complex float alpha, long incx, _Complex float* x);
extern void blas2_sscal(long N, const float* alpha, long incx, float* x);
extern void blas_sscal(long N, float alpha, long incx, float* x);

extern void blas_cdgmm(long M, long N, _Bool left_mul, const _Complex float* A, long lda, const _Complex float* x, long incx, _Complex float* C, long ldc);
extern void blas_sdgmm(long M, long N, _Bool left_mul, const float* A, long lda, const float* x, long incx, float* C, long ldc);

extern void blas2_cdotu(_Complex float* result, long N, long incx, const _Complex float* x, long incy, const _Complex float* y);
extern void blas2_sdot(float* result, long N, long incx, const float* x, long incy, const float* y);

extern void blas_cmatcopy(char trans, long M, long N, _Complex float alpha, const _Complex float* A, long lda, _Complex float* B, long ldb);
extern void blas2_cmatcopy(char trans, long M, long N, const _Complex float* alpha, const _Complex float* A, long lda, _Complex float* B, long ldb);
extern void blas_smatcopy(char trans, long M, long N, float alpha, const float* A, long lda, float* B, long ldb);
extern void blas2_smatcopy(char trans, long M, long N, const float* alpha, const float* A, long lda, float* B, long ldb);

extern void blas_csyrk(char uplow, char trans, long N, long K, _Complex float alpha, long lda, const _Complex float A[*][lda], _Complex float beta, long ldc, _Complex float C[*][ldc]);

extern void blas_matrix_multiply(long M, long N, long K, _Complex float C[N][M], const _Complex float A[K][M], const _Complex float B[N][K]);

extern void blas_matrix_zfmac(long M, long K, long N, _Complex float* C, const _Complex float* A, char transa, const _Complex float* B, char transb);
extern void blas_gemv_zfmac(long M, long N, _Complex float* y, const _Complex float* A, char trans, const _Complex float* x);
extern void blas_gemv_fmac(long M, long N, float* y, const float* A, char trans, const float* x);
extern void blas_sger_fmac(long M, long N, float* A, const float* x, const float* y);

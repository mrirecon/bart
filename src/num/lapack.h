/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#include <complex.h>

#ifdef __cplusplus
#error This file does not support C++
#endif

extern void lapack_eig(long N, float eigenval[N], complex float matrix[N][N]);
extern void lapack_svd(long M, long N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M]);
extern void lapack_svd_econ(long M, long N,
		     complex float U[M][(N > M) ? M : N],
		     complex float VH[(N > M) ? M : N][N],
		     float S[(N > M) ? M : N],
		     complex float A[N][M]);

extern void lapack_eig_double(long N, double eigenval[N], complex double matrix[N][N]);
extern void lapack_svd_double(long M, long N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[N][M]);
extern void lapack_matrix_multiply(long M, long N, long K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N]);
extern void lapack_cholesky(long N, complex float A[N][N]);



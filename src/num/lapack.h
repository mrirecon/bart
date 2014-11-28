/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#include <complex.h>

extern void eigendecomp(long N, float eigenval[N], complex float matrix[N][N]);
extern void svd(long M, long N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], const complex float A[N][M]);
extern void svd_econ(long M, long N, 
		     complex float U[M][(N > M) ? M : N], 
		     complex float VH[(N > M) ? M : N][N],
		     float S[(N > M) ? M : N],
		     const complex float A[N][M]);

extern void eigendecomp_double(long N, double eigenval[N], complex double matrix[N][N]);
extern void svd_double(long M, long N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[N][M]);
extern void matrix_multiply(long M, long N, long K, complex float C[M][N], const complex float A[M][K], const complex float B[K][N]);
extern void cgemm_sameplace(const char transa, const char transb, long M, long N, long K, const complex float* alpha, const complex float A[M][K], const long lda, const complex float B[K][N], const long ldb, const complex float* beta, complex float C[M][N], const long ldc);

#if 1
/* Macro wrappers for functions to work around a limitation of
 * the C language standard: A pointer to an array cannot passed
 * as a pointer to a constant array without adding an explicit cast.
 * We hide this cast in the macro definitions. For GCC we can define
 * a type-safe version of the macro.
 *
 * A similar idea is used in Jens Gustedt's P99 preprocessor macros
 * and functions package available at: http://p99.gforge.inria.fr/
 */
#ifndef AR2D_CAST
#ifndef __GNUC__
#define AR2D_CAST(t, n, m, x) (const t(*)[m])(x)
//#define AR2D_CAST(t, n, m, x) ((const t(*)[m])(0 ? (t(*)[m])0 : (x)))
//#define AR2D_CAST(t, n, m, x) ((const t(*)[m])(t(*)[m]){ &(x[0]) })
#else
#ifndef BUILD_BUG_ON
#define BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2*!!(condition)]))
#endif
#define AR2D_CAST(t, n, m, x) (BUILD_BUG_ON(!(__builtin_types_compatible_p(const t[m], __typeof__((x)[0])) \
				|| __builtin_types_compatible_p(t[m], __typeof__((x)[0])))), (const t(*)[m])(x))
#endif
#endif

#define svd(M, N, U, VH, S, A) \
	svd(M, N, U, VH, S, AR2D_CAST(complex float, N, M, A))

#define svd_econ(M, N, U, VH, S, A) \
	svd_econ(M, N, U, VH, S, AR2D_CAST(complex float, N, M, A))

#endif


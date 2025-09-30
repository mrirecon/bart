/* Copyright 2017. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker
 * 2017 Jon Tamir
 */

#include <complex.h>
#include <stdbool.h>

#include "misc/misc.h"

#ifdef NOLAPACKE
#include "lapacke/lapacke.h"
#elif USE_MKL
#include <mkl.h>
#else
#include <lapacke.h>
#endif

#include "lapack.h"


#define LAPACKE(x, ...) \
	if (0 != LAPACKE_##x(LAPACK_COL_MAJOR, __VA_ARGS__))	\
		error("LAPACK: " # x " failed.\n");

/* ATTENTION: blas and lapack use column-major matrices
 * while native C uses row-major. All matrices are
 * transposed to what one would expect.
 *
 * LAPACK svd destroys its input matrix
 **/

void lapack_eig(long N, float eigenval[N], complex float matrix[N][N])
{
	LAPACKE(cheev, 'V', 'U', N, &matrix[0][0], N, eigenval);
}

// A*x = (lambda)*B*x
void lapack_geig(long N, float eigenval[N], complex float A[N][N], complex float B[N][N])
{
	LAPACKE(chegv, 1, 'V', 'U', N, &A[0][0], N, &B[0][0], N, eigenval);
}

void lapack_svd(long M, long N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M])
{
	LAPACKE(cgesdd, 'A', M, N, &A[0][0], M, S, &U[0][0], M, &VH[0][0], N);
}

// AT = VHT ST UT
void lapack_svd_econ(long M, long N,
		     complex float U[(N > M) ? M : N][M],
		     complex float VH[N][(N > M) ? M : N],
		     float S[(N > M) ? M : N],
		     complex float A[N][M])
{
	PTR_ALLOC(float[MIN(M, N) - 1], superb);
	LAPACKE(cgesvd, NULL != U ? 'S' : 'N', NULL != VH ? 'S' : 'N', M, N, &A[0][0], M, S, NULL != U ? &U[0][0] : NULL, M, NULL != VH ? &VH[0][0] : NULL, MIN(M, N), *superb);
	PTR_FREE(superb);
}

void lapack_eig_double(long N, double eigenval[N], complex double matrix[N][N])
{
	LAPACKE(zheev, 'V', 'U', N, &matrix[0][0], N, eigenval);
}

void lapack_svd_double(long M, long N, complex double U[M][M], complex double VH[N][N], double S[(N > M) ? M : N], complex double A[N][M])
{
	LAPACKE(zgesdd, 'A', M, N, &A[0][0], M, S, &U[0][0], M, &VH[0][0], N);
}

static void lapack_cholesky_UL(long N, char UL, complex float A[N][N])
{
	LAPACKE(cpotrf, UL, N, &A[0][0], N);
}

void lapack_cholesky(long N, complex float A[N][N])
{
	lapack_cholesky_UL(N, 'U', A);
}

void lapack_cholesky_lower(long N, complex float A[N][N])
{
	lapack_cholesky_UL(N, 'L', A);
}


static void lapack_trimat_inverse_UL(long N, char UL, complex float A[N][N])
{
	LAPACKE(ctrtri, UL, 'N', N, &A[0][0], N);
}

void lapack_trimat_inverse(long N, complex float A[N][N])
{
	lapack_trimat_inverse_UL(N, 'U', A);
}

void lapack_trimat_inverse_lower(long N, complex float A[N][N])
{
	lapack_trimat_inverse_UL(N, 'L', A);
}

// Solve A x = B for x
void lapack_trimat_solve(long N, long M, complex float A[N][N], complex float B[M][N], bool upper)
{
	// for non-unit ('N') triangular matrix A
	// on output: B overwritten by solution matrix X
	LAPACKE(ctrtrs, (upper ? 'U' : 'L'), 'N', 'N', N, M, &A[0][0], N, &B[0][0], N);
}


void lapack_cinverse_UL(long N, complex float A[N][N])
{
	int ipiv[N];

	LAPACKE(cgetrf, N, N, &A[0][0], N, ipiv);
	LAPACKE(cgetri, N, &A[0][0], N, ipiv);
}

void lapack_sinverse_UL(long N, float A[N][N])
{
	int ipiv[N];

	LAPACKE(sgetrf, N, N, &A[0][0], N, ipiv);
	LAPACKE(sgetri, N, &A[0][0], N, ipiv);
}


void lapack_schur(long N, complex float W[N], complex float VS[N][N], complex float A[N][N])
{
	int sdim = 0;

	// On output, A overwritten by Schur form T
	LAPACKE(cgees, 'V', 'N', NULL, N, &A[0][0], N, &sdim, &W[0], &VS[0][0], N);
}

void lapack_schur_double(long N, complex double W[N], complex double VS[N][N], complex double A[N][N])
{
	int sdim = 0;

	// On output, A overwritten by Schur form T
	LAPACKE(zgees, 'V', 'N', NULL, N, &A[0][0], N, &sdim, &W[0], &VS[0][0], N);
}

// Solves the complex Sylvester matrix equation
// op(A)*X + X*op(B) = scale*C
void lapack_sylvester(long N, long M, float* scale, complex float A[N][N], complex float B[M][M], complex float C[M][N])
{
	// A -> triangluar
	// On output: C overwritten by X
	LAPACKE(ctrsyl, 'N', 'N', +1, N, M, &A[0][0], N, &B[0][0], M, &C[0][0], N, scale);
}

void lapack_solve_real(long N, float A[N][N], float B[N])
{
	int ipiv[N];
	LAPACKE(sgesv, N, 1, &A[0][0], N, ipiv, B, N);
}



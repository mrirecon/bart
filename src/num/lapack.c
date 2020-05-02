/* Copyright 2017. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>

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
		error("LAPACK: " # x " failed.");

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

void lapack_svd(long M, long N, complex float U[M][M], complex float VH[N][N], float S[(N > M) ? M : N], complex float A[N][M])
{
	LAPACKE(cgesdd, 'A', M, N, &A[0][0], M, S, &U[0][0], M, &VH[0][0], N);
}

void lapack_svd_econ(long M, long N,
		     complex float U[M][(N > M) ? M : N],
		     complex float VH[(N > M) ? M : N][N],
		     float S[(N > M) ? M : N],
		     complex float A[N][M])
{
	PTR_ALLOC(float[MIN(M, N) - 1], superb);
	LAPACKE(cgesvd, 'S', 'S', M, N, &A[0][0], M, S, &U[0][0], M, &VH[0][0], MIN(M, N), *superb);
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





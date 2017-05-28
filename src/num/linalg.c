/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 *
 *
 * Simple linear algebra functions.
 */

#include <complex.h>
#include <math.h>
#include <assert.h>

#if 1
// #define MAT_USE_LAPACK
#define DOUBLE_ACC
#endif

#include "misc/misc.h"

#ifdef MAT_USE_LAPACK
#include "num/blas.h"
#include "num/lapack.h"
#endif
#include "num/rand.h"

#include "linalg.h"



#ifdef DOUBLE_ACC
typedef complex double cfl_acu_t;
typedef double fl_acu_t;
#else
typedef complex float cfl_acu_t;
typedef float fl_acu_t;
#endif



void mat_identity(int A, int B, complex float x[A][B])
{
	for (int i = 0; i < A; i++)
		for (int j = 0; j < B; j++)
			x[i][j] = (i == j) ? 1. : 0.;
}

void mat_zero(int A, int B, complex float m[A][B])
{
	for (int a = 0; a < A; a++)
		for (int b = 0; b < B; b++)
			m[a][b] = 0.;
}

void mat_gaussian(int A, int B, complex float x[A][B])
{
	for (int i = 0; i < A; i++)
		for (int j = 0; j < B; j++)
			x[i][j] = gaussian_rand();
}

// add constant to vector
void vec_sadd(long D, complex float alpha, complex float dst[D], const complex float src[D])
{
//	#pragma omp parallel for
	for (long i = 0; i < D; i++)
		dst[i] = alpha + src[i];
}

complex float vec_mean(long D, const complex float src[D])
{
	cfl_acu_t val = 0;

	for (long i = 0; i < D; i++)
		val += src[i];

	return val / D;
}

void (mat_add)(int A, int B, complex float x[A][B], const complex float y[A][B], const complex float z[A][B])
{
	for (int i = 0; i < A; i++)
		for (int j = 0; j < B; j++)
			x[i][j] = y[i][j] + z[i][j];
}

void (mat_muladd)(int A, int B, int C, complex float x[MVLA(A)][C], const complex float y[MVLA(A)][B], const complex float z[MVLA(B)][C])
{
#ifdef MAT_USE_LAPACK
	complex float tmp[A][C];
	mat_mul(A, B, C, tmp, y, z);
	mat_add(A, C, x, x, tmp);
#else
	for (int i = 0; i < A; i++) {
		for (int j = 0; j < C; j++) {

			cfl_acu_t tmp = 0.;

			for (int k = 0; k < B; k++)
				tmp += y[i][k] * z[k][j];

			x[i][j] += tmp;
		}
	}
#endif
}


void (mat_mul)(int A, int B, int C, complex float x[A][C], const complex float y[A][B], const complex float z[B][C])
{
#ifdef MAT_USE_LAPACK
	blas_matrix_multiply(C, A, B, x, z, y);
#else
	for (int i = 0; i < A; i++) {
		for (int j = 0; j < C; j++) {

			cfl_acu_t tmp = 0.;

			for (int k = 0; k < B; k++)
				tmp += y[i][k] * z[k][j];

			x[i][j] = tmp;
		}
	}
#endif
}


bool (mat_inverse)(unsigned int N, complex float out[N][N], const complex float in[N][N])
{
#ifdef MAT_USE_LAPACK
//	return blas_matrix_inverse(N, out, in);
	UNUSED(in);
	UNUSED(out);
	assert(0);
#else
	// ATTENTION: slow and inaccurate

	complex float tmp[2 * N][N];
	mat_transpose(N, N, tmp, in);
	mat_identity(N, N, tmp + N);

	complex float tmp2[N][2 * N];
	mat_transpose(2 * N, N, tmp2, tmp);

	for (unsigned int i = 0; i < N; i++) {

		complex float diag = tmp2[i][i];

		if (0. == diag)
			return false;

		for (unsigned int j = 0; j < 2 * N; j++)
			tmp2[i][j] /= diag;

		for (unsigned int j = 0; j < N; j++) {

			if (i != j)
				vec_saxpy(2 * N, tmp2[j], -tmp2[j][i], tmp2[i]);
		}
	}

	mat_transpose(N, 2 * N, tmp, tmp2);
	mat_transpose(N, N, out, tmp + N);

	return true;
#endif
}

void (mat_kron)(unsigned int A, unsigned int B, unsigned int C, unsigned int D, 
		complex float out[A * C][B * D], const complex float in1[A][B], const complex float in2[C][D])
{
	for (unsigned int a = 0; a < A; a++)
		for (unsigned int b = 0; b < B; b++)
			for (unsigned int c = 0; c < C; c++)
				for (unsigned int d = 0; d < D; d++)
					out[a + c * A][b + d * B] = in1[a][b] * in2[c][d];
}


void (mat_vecmul)(unsigned int A, unsigned int B, complex float out[A], const complex float mat[A][B], const complex float in[B])
{
	for (unsigned int a = 0; a < A; a++) {

		cfl_acu_t tmp = 0.;

		for (unsigned int b = 0; b < B; b++)
			tmp += mat[a][b] * in[b];

		out[a] = tmp;
	}
}

void (mat_vec)(unsigned int A, unsigned int B, complex float out[A * B], const complex float in[A][B])
{
	for (unsigned int a = 0; a < A; a++)
		for (unsigned int b = 0; b < B; b++)
			out[a * B + b] = in[a][b];
}

void (vec_mat)(unsigned int A, unsigned int B, complex float out[A][B], const complex float in[A * B])
{
	for (unsigned int a = 0; a < A; a++)
		for (unsigned int b = 0; b < B; b++)
			out[a][b] = in[a * B + b];
}


complex float vec_dot(int N, const complex float x[N], const complex float y[N])
{
	cfl_acu_t scalar = 0.;

	// use double here to avoid errors
	// one could also look into the Kahan summation algorithm

	for (int k = 0; k < N; k++)
		scalar += x[k] * conjf(y[k]);

	return scalar;	
}


// FIXME: this is not axpy
void vec_axpy(long N, complex float x[N], complex float alpha, const complex float y[N])
{
//	#pragma omp parallel for
	for (long k = 0; k < N; k++)
		x[k] = alpha * y[k];
}

void vec_saxpy(int N, complex float x[N], complex float alpha, const complex float y[N])
{
	for (int k = 0; k < N; k++)
		x[k] += alpha * y[k];
}

void (gram_matrix)(int N, complex float cov[N][N], int L, const complex float data[N][L])
{
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		for (int j = 0; j <= i; j++) {	

			complex float val = vec_dot(L, data[i], data[j]);

			cov[j][i] = val;
			cov[i][j] = conj(val);
		}
	}
}

void (pack_tri_matrix)(int N, complex float cov[N * (N + 1) / 2], const complex float m[N][N])
{
	int l = 0;

	for (int i = 0; i < N; i++)
		for (int j = 0; j <= i; j++)
			cov[l++] = m[i][j];
}

void (unpack_tri_matrix)(int N, complex float m[N][N], const complex float cov[N * (N + 1) / 2])
{
	int l = 0;

	for (int i = 0; i < N; i++)
		for (int j = 0; j <= i; j++)
			m[i][j] = cov[l++];
}

void (gram_matrix2)(int N, complex float cov[N * (N + 1) / 2], int L, const complex float data[N][L])
{
#if 0
	int l = 0;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j <= i; j++) {	

			complex float val = vec_dot(L, data[i], data[j]);

			cov[l++] = conj(val);
		}
	}
#else
	complex float c[N][N];
	gram_matrix(N, c, L, data);
	pack_tri_matrix(N, cov, c);
#endif
}





void gram_schmidt(int M, int N, float vals[M], complex float vecs[M][N])
{
	if (M > 1)
		gram_schmidt(M - 1, N, vals + 1, vecs + 1);

	for (int j = 1; j < M; j++) {

		complex float scalar = vec_dot(N, vecs[0], vecs[j]);

		vec_saxpy(N, vecs[0], -scalar, vecs[j]);
	}

	vals[0] = sqrtf(crealf(vec_dot(N, vecs[0], vecs[0])));

	for (int k = 0; k < N; k++)
		vecs[0][k] /= vals[0];
}

void (mat_transpose)(int A, int B, complex float dst[B][A], const complex float src[A][B])
{
	for (int i = 0; i < B; i++)
		for (int j = 0; j < A; j++)
			dst[i][j] = src[j][i];	// swap
}

void (mat_adjoint)(int A, int B, complex float dst[B][A], const complex float src[A][B])
{
	for (int i = 0; i < B; i++)
		for (int j = 0; j < A; j++)
			dst[i][j] = conjf(src[j][i]);	// swap
}

void (mat_copy)(int A, int B, complex float dst[A][B], const complex float src[A][B])
{
	for (int i = 0; i < A; i++)
		for (int j = 0; j < B; j++)
			dst[i][j] = src[i][j];
}

void (orthiter_noinit)(int M, int N, int iter, float val[M], complex float out[M][N], const complex float matrix[N][N])
{
	complex float tmp[M][N];

	for (int n = 0; n < iter; n++) {

		mat_copy(M, N, tmp, out);
		mat_mul(M, N, N, out, tmp, matrix);
		gram_schmidt(M, N, val, out); 
	}
}


void (orthiter)(int M, int N, int iter, float val[M], complex float out[M][N], const complex float matrix[N][N])
{
	mat_identity(M, N, out);
	orthiter_noinit(M, N, iter, val, out, matrix);
}


void cholesky_double(int N, complex double A[N][N])
{
        for (int i = 0; i < N; i++) {

                for (int j = 0; j < i; j++) {

                        cfl_acu_t sum = A[i][j];

                        for (int k = 0; k < j; k++)
                                sum -= A[i][k] * conj(A[j][k]); 

                        A[i][j] = sum / A[j][j];
                }

                fl_acu_t sum = creal(A[i][i]);

                for (int k = 0; k < i; k++)
                        sum -= creal(A[i][k] * conj(A[i][k]));

            	assert(sum > 0.);

                A[i][i] = sqrt(sum);
        }

        for (int i = 0; i < N; i++)
                for (int j = 0; j < i; j++)
			A[j][i] = conj(A[i][j]);
	
}

// Tadeusz Banachiewicz
void cholesky(int N, complex float A[N][N])
{
#ifdef MAT_USE_LAPACK
	lapack_cholesky(N, A);

        for (int i = 0; i < N; i++)
                for (int j = 0; j < i; j++)
			A[j][i] = conjf(A[i][j]);
#else
#if 0
	complex double B[N][N];

        for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) //
			B[i][j] = A[i][j];
	
	cholesky_double(N, B);

        for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++) // 
			A[i][j] = B[i][j];
#else
        for (int i = 0; i < N; i++) {

                for (int j = 0; j < i; j++) {

                        cfl_acu_t sum = A[i][j];

                        for (int k = 0; k < j; k++)
                                sum -= A[i][k] * conjf(A[j][k]);

                        A[i][j] = sum / A[j][j];
                }

                fl_acu_t sum = creal(A[i][i]);

                for (int k = 0; k < i; k++)
                        sum -= crealf(A[i][k] * conjf(A[i][k]));

                assert(sum > 0.);

                A[i][i] = sqrt(sum);
        }

        for (int i = 0; i < N; i++)
                for (int j = 0; j < i; j++)
			A[j][i] = conjf(A[i][j]);
#endif
#endif
}

#if 0
static void backsubst_lower_double(int N, complex double x[N], complex double L[N][N], complex double b[N])
{
	for (int i = 0; i < N; i++) {

		complex double sum = b[i];

		for (int j = 0; j < i; j++)
			sum -= x[j] * L[i][j];
		
		x[i] = sum / L[i][i];
	}
}

static void backsubst_upper_double(int N, complex double x[N], complex double L[N][N], complex double b[N])
{
	for (int i = N - 1; i >= 0; i--) {

		complex double sum = b[i];

		for (int j = i + 1; j < N; j++)
			sum -= x[j] * L[i][j];
		
		x[i] = sum / L[i][i];
	}
}

void mat_adjoint_double(int A, int B, complex double dst[B][A], complex double src[A][B])
{
	for (int i = 0; i < B; i++)
		for (int j = 0; j < A; j++)
			dst[i][j] = conj(src[j][i]);	// swap
}

void cholesky_solve_double(int N, complex double x[N], complex double L[N][N], complex double b[N])
{
	complex double y[N];
	complex double T[N][N];

	mat_adjoint_double(N, N, T, L);

	backsubst_lower_double(N, y, L, b);
	backsubst_upper_double(N, x, T, y);
}

#endif



static void backsubst_lower(int N, complex float x[N], const complex float L[N][N], const complex float b[N])
{
	for (int i = 0; i < N; i++) {

		cfl_acu_t sum = b[i];

		for (int j = 0; j < i; j++)
			sum -= x[j] * L[j][i];
		
		x[i] = sum / L[i][i];
	}
}

static void backsubst_upper(int N, complex float x[N], const complex float L[N][N], const complex float b[N])
{
	for (int i = N - 1; i >= 0; i--) {

		cfl_acu_t sum = b[i];

		for (int j = i + 1; j < N; j++)
			sum -= x[j] * L[j][i];
		
		x[i] = sum / L[i][i];
	}
}



void (cholesky_solve)(int N, complex float x[N], const complex float L[N][N], const complex float b[N])
{
	complex float y[N];
	
	backsubst_lower(N, y, L, b);
	backsubst_upper(N, x, L, y);
}



void thomas_algorithm(int N, complex float f[N], const complex float A[N][3], const complex float d[N])
{
	complex float c[N];
	complex float e[N];

	c[0] = A[0][2] / A[0][1];
	e[0] = d[0] / A[0][1];

	for (int i = 1; i < N; i++) {

		c[i] = A[i][2] / (A[i][1] - c[i - 1] * A[i][0]);
		e[i] = (d[i] - A[i][0] * e[i - 1]) / (A[i][1] - A[i][0] * c[i - 1]);
	}

	// backsubstitution

	f[N - 1] = e[N - 1];

	for (int i = N - 2; 0 <= i; i--)
		f[i] = e[i] - c[i] * f[i + 1];
}


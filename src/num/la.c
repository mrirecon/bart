/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com>
 *
 *
 * Simple linear algebra functions.
 */

#include <complex.h>
#include <math.h>
#include <assert.h>

#include "num/rand.h"

#include "la.h"



void mat_identity(int A, int B, complex float x[A][B])
{
	for (int i = 0; i < A; i++)
		for (int j = 0; j < B; j++)
			x[i][j] = (i == j) ? 1. : 0.;
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
	#pragma omp parallel for
	for (long i = 0; i < D; i++)
		dst[i] = alpha + src[i];
}

complex float vec_mean(long D, const complex float src[D])
{
	complex double val = 0;

	for (long i = 0; i < D; i++)
		val += src[i];

	return val / D;
}


void (mat_mul)(int A, int B, int C, complex float x[A][C], const complex float y[A][B], const complex float z[B][C])
{
	for (int i = 0; i < A; i++) {
		for (int j = 0; j < C; j++) {

			//complex double tmp = 0.;
			complex float tmp = 0.;

			for (int k = 0; k < B; k++)
				tmp += y[i][k] * z[k][j];

			x[i][j] = tmp;
		}
	}
}


complex float vec_dot(int N, const complex float x[N], const complex float y[N])
{
	complex double scalar = 0.;

	// use double here to avoid errors
	// one could also look into the Kahan summation algorithm

	for (int k = 0; k < N; k++)
		scalar += x[k] * conjf(y[k]);

	return scalar;	
}


// FIXME: this is not axpy
void vec_axpy(long N, complex float x[N], complex float alpha, const complex float y[N])
{
	#pragma omp parallel for
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


void (orthiter)(int M, int N, int iter, float val[M], complex float out[M][N], const complex float matrix[N][N])
{
	complex float tmp[M][N];

	mat_identity(M, N, out);

	for (int n = 0; n < iter; n++) {

		mat_copy(M, N, tmp, out);
		mat_mul(M, N, N, out, tmp, matrix);
		gram_schmidt(M, N, val, out); 
	}
}

void cholesky_double(int N, complex double A[N][N])
{
        for (int i = 0; i < N; i++) {

                for (int j = 0; j < i; j++) {

                        complex double sum = A[i][j];

                        for (int k = 0; k < j; k++)
                                sum -= A[i][k] * conj(A[j][k]); 

                        A[i][j] = sum / A[j][j];
                }

                double sum = creal(A[i][i]);

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

                        complex double sum = A[i][j];

                        for (int k = 0; k < j; k++)
                                sum -= A[i][k] * conjf(A[j][k]);

                        A[i][j] = sum / A[j][j];
                }

                double sum = creal(A[i][i]);

                for (int k = 0; k < i; k++)
                        sum -= crealf(A[i][k] * conjf(A[i][k]));

                assert(sum > 0.);

                A[i][i] = sqrt(sum);
        }

        for (int i = 0; i < N; i++)
                for (int j = 0; j < i; j++)
			A[j][i] = conjf(A[i][j]);
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

		complex double sum = b[i];

		for (int j = 0; j < i; j++)
			sum -= x[j] * L[j][i];
		
		x[i] = sum / L[i][i];
	}
}

static void backsubst_upper(int N, complex float x[N], const complex float L[N][N], const complex float b[N])
{
	for (int i = N - 1; i >= 0; i--) {

		complex double sum = b[i];

		for (int j = i + 1; j < N; j++)
			sum -= x[j] * L[j][i];
		
		x[i] = sum / L[i][i];
	}
}



void cholesky_solve(int N, complex float x[N], const complex float L[N][N], const complex float b[N])
{
	complex float y[N];
	
	backsubst_lower(N, y, L, b);
	backsubst_upper(N, x, L, y);
}






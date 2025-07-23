/* Copyright 2015. The Regents of the University of California.
 * Copyright 2016-2020. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024      Nick Scholand
 * 2019-2020 Sebastian Rosenzweig
 * 2012-2020 Martin Uecker
 * 2013	     Dara Bahri
 *
 *
 * Simple linear algebra functions.
 */

#include <complex.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>

#if 1
// #define MAT_USE_LAPACK
#define DOUBLE_ACC
#endif

#include "misc/misc.h"

#include "misc/debug.h"

#ifdef MAT_USE_LAPACK
#include "num/blas.h"
#endif
#include "num/lapack.h"
#include "num/rand.h"
#include "num/specfun.h"

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

void vec_zero(int N, complex float x[N])
{
	for (int i = 0; i < N; i++)
		x[i] = 0.;
}

void vec_copy(int N, complex float x[N], const complex float y[N])
{
	for (int i = 0; i < N; i++)
		x[i] = y[i];
}

void vecf_copy(int N, float x[N], const float y[N])
{
	for (int i = 0; i < N; i++)
		x[i] = y[i];
}

void vecf_saxpy(int N, float dst[N], float alpha, const float b[N])
{
	for (int i = 0; i < N; i++)
		dst[i] += alpha * b[i];
}

float vecf_sdot(int N, const float a[N], const float b[N])
{
	float ret = 0.;

	for (int i = 0; i < N; i++)
		ret += a[i] * b[i];

	return ret;
}

float vecf_norm(int N, const float x[N])
{
	return sqrtf(vecf_sdot(N, x, x));
}

#ifndef NO_LAPACK
void matf_solve(int N, float x[N], const float m[N][N], const float y[N])
{
	float tmp[N][N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			tmp[i][j] = m[j][i];	// transpose

	for (int i = 0; i < N; i++)
		x[i] = y[i];

	lapack_solve_real(N, tmp, x);
}
#endif

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

	return val / (float)D;
}

void mat_add(int A, int B, complex float x[A][B], const complex float y[A][B], const complex float z[A][B])
{
	for (int i = 0; i < A; i++)
		for (int j = 0; j < B; j++)
			x[i][j] = y[i][j] + z[i][j];
}

void mat_muladd(int A, int B, int C, complex float x[MVLA(A)][C], const complex float y[MVLA(A)][B], const complex float z[MVLA(B)][C])
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


void mat_mul(int A, int B, int C, complex float x[A][C], const complex float y[A][B], const complex float z[B][C])
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


complex float mat_det(int N, const complex float mat[N][N])
{
	assert(N <= 2);

	if (0 == N)
		return 0.;

	if (1 == N)
		return mat[0][0];

	if (2 == N)
		return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];

	return 0.;
}


void matf_mul(int A, int B, int C, float x[A][C], const float y[A][B], const float z[B][C])
{
	for (int i = 0; i < A; i++) {
		for (int j = 0; j < C; j++) {

			fl_acu_t tmp = 0.;

			for (int k = 0; k < B; k++)
				tmp += y[i][k] * z[k][j];

			x[i][j] = tmp;
		}
	}
}



bool mat_inverse(int N, complex float out[N][N], const complex float in[N][N])
{
#ifdef MAT_USE_LAPACK
//	return blas_matrix_inverse(N, out, in);
	(void)in;
	(void)out;
	assert(0);
#else
	// ATTENTION: slow and inaccurate

	complex float tmp[2 * N][N];
	mat_transpose(N, N, tmp, in);
	mat_identity(N, N, tmp + N);

	complex float tmp2[N][2 * N];
	mat_transpose(2 * N, N, tmp2, tmp);

	for (int i = 0; i < N; i++) {

		complex float diag = tmp2[i][i];

		if (0. == diag)
			return false;

		for (int j = 0; j < 2 * N; j++)
			tmp2[i][j] /= diag;

		for (int j = 0; j < N; j++) {

			if (i != j)
				vec_saxpy(2 * N, tmp2[j], -tmp2[j][i], tmp2[i]);
		}
	}

	mat_transpose(N, 2 * N, tmp, tmp2);
	mat_transpose(N, N, out, tmp + N);

	return true;
#endif
}


// Moore-Penrose pseudo inverse
void mat_pinv(int A, int B, complex float out[B][A], const complex float in[A][B])
{
	((B <= A) ? mat_pinv_left : mat_pinv_right)(A, B, out, in);
}


void mat_pinv_left(int A, int B, complex float out[B][A], const complex float in[A][B])
{
	if (A == B) {

		mat_inverse(A, out, in);
		return;
	}

	assert(B < A);

	complex float adj[B][A];
	mat_adjoint(A, B, adj, in);

	complex float prod[B][B];
	mat_mul(B, A, B, prod, adj, in);

	complex float inv[B][B];
	mat_inverse(B, inv, prod);

	mat_mul(B, B, A, out, inv, adj);
}


void mat_pinv_right(int A, int B, complex float out[B][A], const complex float in[A][B])
{
	if (A == B) {

		mat_inverse(A, out, in);
		return;
	}

	assert(A < B);

	complex float adj[B][A];
	mat_adjoint(A, B, adj, in);

	complex float prod[A][A];
	mat_mul(A, B, A, prod, in, adj);

	complex float inv[A][A];
	mat_inverse(A, inv, prod);

	mat_mul(B, A, A, out, adj, inv);
}

static void mat_vecmul_columnwise(int A, int B, complex float out[A][B], const complex float mat[A][B], const float in[A])
{
	for (int a = 0; a < A; a++)
		for (int b = 0; b < B; b++)
			out[a][b] = mat[a][b] * in[a];
}

void mat_svd_recov(int A, int B, complex float out[A][B], const complex float U[A][A], const complex float VH[B][B], const float S[A])
{
	complex float VH2[B][B];

	mat_vecmul_columnwise(((B > A) ? A : B), B, VH2, VH, S);

	mat_mul(A, A, B, out, U, VH2);
}

#ifndef NO_LAPACK
// Wrapper for lapack including row-major definition of svd
void mat_svd(int A, int B, complex float U[A][A], complex float VH[B][B], float S[A], const complex float in[A][B])
{
	// Avoid overwriting "in" by lapack call
	complex float in2[A][B];
	mat_copy(A, B, in2, in);

	lapack_svd(B, A, VH, U, S, in2);
}

// pinv(in) = V S^{-1} U^T
void mat_pinv_svd(int A, int B, complex float out[B][A], const complex float in1[A][B])
{
	// Take conj transpose for complex into account
	complex float in[A][B];
	mat_conj(A, B, in, in1);

	complex float VH[B][B];
	complex float U[A][A];
	float S[A];

	//  U S V^H = in
	mat_svd(A, B, U, VH, S, in);

	// S^{-1}
	float tol = 1e-6;

	for (int i = 0; i < A; i++)
		if (tol < S[i])
			S[i] = 1. / S[i];

	// UT = U^T
	complex float UT[A][A];
	mat_transpose(A, A, UT, U);

	// V = VH^T
	complex float V[B][B];
	mat_transpose(B, B, V, VH);

	// U2 = S^{-1} UT -> consider S is diagonal matrix!
	complex float U2[A][A];
	mat_vecmul_columnwise(A, A, U2, UT, S);

	// FIXME: Avoid cutting by moving to lapack_svd_econ?
	if (A <= B) {

		complex float V_cut[B][A];

		for (int i = 0; i < B; i++)
			for (int j = 0; j < A; j++)
				V_cut[i][j] = V[i][j];

		// pinv(in) = (V^H)^T U2
		mat_mul(B, A, A, out, V_cut, U2);

	} else {

		complex float U_cut[B][A];

		for (int i = 0; i < B; i++)
			for (int j = 0; j < A; j++)
				U_cut[i][j] = U2[i][j];

		// pinv(in) = (V^H)^T U2
		mat_mul(B, B, A, out, V, U_cut);
	}
}

void mat_schur_recov(int A, complex float out[A][A], const complex float T[A][A], const complex float Z[A][A])
{

	complex float Z_adj[A][A];
	mat_adjoint(A, A, Z_adj, Z);

	complex float T2[A][A];
	mat_mul(A, A, A, T2, Z, T);

	mat_mul(A, A, A, out, T2, Z_adj);
}


void mat_schur(int A, complex float T[A][A], complex float Z[A][A], const complex float in[A][A])
{
	// transpose -> lapack use column-major matrices while native C uses row-major
	complex float EV[A];
	complex float T2[A][A];
	complex float Z2[A][A];

	mat_transpose(A, A, T2, in);

	lapack_schur(A, EV, Z2, T2);

	mat_transpose(A, A, Z, Z2);
	mat_transpose(A, A, T, T2);
}


void mat_ceig_double(int A, complex double EV[A], const complex double in[A][A])
{
	complex double tmp[A][A];

	// transpose -> lapack use column-major matrices while native C uses row-major
	for (int i = 0; i < A; i++)
		for (int j = 0; j < A; j++)
			tmp[i][j] = in[j][i];

	complex double vec[A][A];

	lapack_schur_double(A, EV, vec, tmp);
}

void mat_eig_double(int A, double EV[A], const double in[A][A])
{
	complex double tmp[A][A];
	complex double tmp2[A];

	// transpose -> lapack use column-major matrices while native C uses row-major
	for (int i = 0; i < A; i++)
		for (int j = 0; j < A; j++)
			tmp[i][j] = in[j][i] + 0.i;

	complex double vec[A][A];
	lapack_schur_double(A, tmp2, vec, tmp);

	for (int i = 0; i < A; i++)
		EV[i] = creal(tmp2[i]);
}
#endif


void mat_kron(int A, int B, int C, int D,
	      complex float out[A * C][B * D], const complex float in1[A][B], const complex float in2[C][D])
{
	for (int a = 0; a < A; a++)
		for (int b = 0; b < B; b++)
			for (int c = 0; c < C; c++)
				for (int d = 0; d < D; d++)
					out[a + c * A][b + d * B] = in1[a][b] * in2[c][d];
}


void mat_vecmul(int A, int B, complex float out[A], const complex float mat[A][B], const complex float in[B])
{
	for (int a = 0; a < A; a++) {

		cfl_acu_t tmp = 0.;

		for (int b = 0; b < B; b++)
			tmp += mat[a][b] * in[b];

		out[a] = tmp;
	}
}


void matf_vecmul(int A, int B, float out[A], const float mat[A][B], const float in[B])
{
	for (int a = 0; a < A; a++) {

		fl_acu_t tmp = 0.;

		for (int b = 0; b < B; b++)
			tmp += mat[a][b] * in[b];

		out[a] = tmp;
	}
}


void mat_vec(int A, int B, complex float out[A * B], const complex float in[A][B])
{
	for (int a = 0; a < A; a++)
		for (int b = 0; b < B; b++)
			out[a * B + b] = in[a][b];
}

void vec_mat(int A, int B, complex float out[A][B], const complex float in[A * B])
{
	for (int a = 0; a < A; a++)
		for (int b = 0; b < B; b++)
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

void gram_matrix(int N, complex float cov[N][N], int L, const complex float data[N][L])
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

void pack_tri_matrix(int N, complex float cov[N * (N + 1) / 2], const complex float m[N][N])
{
	int l = 0;

	for (int i = 0; i < N; i++)
		for (int j = 0; j <= i; j++)
			cov[l++] = m[i][j];
}

#ifndef NO_LAPACK
// Solve M x = N for x with non-unit triangular matrix M
void solve_tri_matrix(int A, int B, complex float M[A][A], complex float N[A][B], bool upper)
{
	// transpose -> lapack use column-major matrices while native C uses row-major
	complex float M2[A][A];
	complex float N2[B][A];

	mat_transpose(A, A, M2, M);
	mat_transpose(A, B, N2, N);

	lapack_trimat_solve(A, B, M2, N2, upper);

	mat_transpose(B, A, N, N2); // Output: N
}

void solve_tri_matrix_vec(int A, complex float M[A][A], complex float N[A], bool upper)
{
	complex float N2[A][1];

	for (int i = 0; i < A; i++)
		N2[i][0] = N[i];

	solve_tri_matrix(A, 1, M, N2, upper);

	for (int i = 0; i < A; i++)
		N[i] = N2[i][0];
}

// Wrapper to solves the complex Sylvester matrix equation
// op(M)*X + X*op(N) = scale*C
void solve_tri_matrix_sylvester(int A, int B, float* scale, complex float M[A][A], complex float N[B][B], complex float C[A][B])
{
	// transpose -> lapack use column-major matrices while native C uses row-major
	complex float M2[A][A];
	complex float N2[B][B];
	complex float C2[B][A];

	mat_transpose(A, A, M2, M);
	mat_transpose(B, B, N2, N);
	mat_transpose(A, B, C2, C);

	lapack_sylvester(A, B, scale, M2, N2, C2);

	mat_transpose(B, A, C, C2); // Output: C
}

// Matrix square root of upper triangular matrix
//	E. Deadman, N. J. Higham, R. Ralha,
//	"Blocked Schur Algorithms for Computing the Matrix Square Root"
//	Lecture Notes in Computer Science, 2013.
void sqrtm_tri_matrix(int N, int blocksize, complex float out[N][N], const complex float in[N][N])
{
	complex float T_diag[N][N];

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {

			T_diag[i][j] = (i == j) ? in[i][j] : 0.;

			out[i][j] = csqrtf(T_diag[i][j]);
		}
	}

	// Implemented standard block method for increased efficiency

	// Number of blocks
	int n_blocks = (1 > N / blocksize) ? 1 : N / blocksize;

	// Sizes of blocks
	int size_small = N / n_blocks;
	int size_large = size_small + 1;
	int n_large = N % n_blocks;
	int n_small = n_blocks - n_large;
	assert(N == n_large * size_large + n_small * size_small);

	// Index ranges
	int s = 0;
	int pairs[n_small + n_large][2];

	// 1. Small
	for (int i = 0; i < n_small; i++) {

		pairs[i][0] = s;
		pairs[i][1] = s + size_small;
		s += size_small;
	}

	// 2. Large
	for (int i = 0; i < n_large; i++) {

		pairs[i + n_small][0] = s;
		pairs[i + n_small][1] = s + size_large;
		s += size_large;
	}

	// Within-Blocks interaction
	for (int i = 0; i < (n_small + n_large); i++) {

		for (int j = pairs[i][0]; j < pairs[i][1]; j++) {

			for (int k = j - 1; k > (pairs[i][0] - 1); k--) {

				complex float s = 0.;

				if (1 < j - k)
					for (int m = k + 1; m < j; m++)
						s += out[k][m] * out[m][j];

				complex float denom = out[k][k] + out[j][j];

				complex float num = in[k][j] - s;

				if (0 != denom)
					out[k][j] = (in[k][j] - s) / denom;
				else if ( (0 == denom) && (0 == num) ) // FIXME: eps
					out[k][j] = 0.;
				else
					error("Error in calculating matrix sqrt of triangular matrix.");
			}
		}
	}

	// Between-Blocks interaction
	for (int i = 0; i < n_blocks; i++) {

		int i_ind_s = pairs[i][0];
		int i_ind_e = pairs[i][1];
		int i_size = i_ind_e - i_ind_s;

		for (int j = i - 1; j > -1; j--) {

			int j_ind_s = pairs[j][0];
			int j_ind_e = pairs[j][1];
			int j_size = j_ind_e - j_ind_s;

			complex float S[j_size][i_size];

			for (int ii = 0; ii < i_size; ii++)
				for (int jj = 0; jj < j_size; jj++)
					S[jj][ii] = in[jj + j_ind_s][ii + i_ind_s];

			if (1 < i - j)
				for (int ii = 0; ii < i_size; ii++)
					for (int jj = 0; jj < j_size; jj++)
						S[jj][ii] = S[jj][ii]
							- out[jj+j_ind_s][ii+j_ind_e]
							* out[jj+j_ind_e][ii+i_ind_s];

			complex float Ujj[j_size][j_size];

			for (int x = 0; x < j_size; x++)
				for (int y = 0; y < j_size; y++)
					Ujj[x][y] = out[x + j_ind_s][y + j_ind_s];

			complex float Uii[i_size][i_size];

			for (int x = 0; x < i_size; x++)
				for (int y = 0; y < i_size; y++)
					Uii[x][y] = out[x + i_ind_s][y + i_ind_s];

			// Solve Sylvester equations for upper triangular matrix Ujj
			float scale = 1.;

			solve_tri_matrix_sylvester(j_size, i_size, &scale, Ujj, Uii, S);

			for (int j = 0; j < j_size; j++)
				for (int ii = 0; ii < i_size; ii++)
					out[j + j_ind_s][ii + i_ind_s] = S[j][ii] * scale;
		}
	}
}

static void matf_sqrt(int N, int M, complex float out[N][M], complex float in[N][M])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			out[i][j] = csqrtf(in[i][j]);
}

static float max_abs_diag(int N, complex float A[N][N])
{
	float max = 0.;

	for (int i = 0; i < N; i++)
		max = (max < cabsf(A[i][i] - 1.f)) ? cabsf(A[i][i] - 1.f) : max;

	return max;
}

// Estimate 1-norm of (A - I)^order
float mat_onenorm_power(int N, int order, complex float A[N][N])
{
	// A - I
	complex float A2[N][N];
	complex float A3[N][N];

	for (int i = 0; i < N; i++) {

		for (int j = 0; j < N; j++) {

			A2[i][j] = (i == j) ? A[i][j] - 1. : A[i][j];
			A3[i][j] = A2[i][j];
		}
	}

	// (A - I)^M
	complex float A4[N][N];

	for (int i = 1; i < order; i++) {

		mat_mul(N, N, N, A4, A2, A3);
		mat_copy(N, N, A3, A4);
	}

	// 1-norm
	float norm = 0.;

	for (int i = 0; i < N; i++) {

		float abs_sum = 0;

		for (int j = 0; j < N; j++)
			abs_sum += cabsf(A3[j][i]);

		norm = (norm < abs_sum) ? abs_sum : norm;
	}

	return norm;
}

// Avoid subtractive cancellation when calculating: out = z^{1/(2^N)}-1
//	A. H. Al-Mohy,
//	"A more accurate Briggs method for the logarithm",
//	Numerical Algorithms, 2012.
static complex float mat_briggs(int N, complex float z)
{
	assert(0 <= N);

	if (0 == N)
		return z - 1.;

	if (1 == N)
		return csqrtf(z) - 1.;

	int N2 = N;

	if (M_PI / 2. <= cargf(z)) {

		z = csqrtf(z);
		N2 = N - 1;
	}

	complex float z0 = z - 1.;

	z = csqrtf(z);

	complex float out = 1 + z;

	for (int i = 1; i < N2; i++) {

		z = csqrtf(z);
		out *= (1. + z);
	}

	out = z0 / out;

	return out;
}

// Superdiagonal of fractional matrix power
// 	N. J. Higham and L. Lin,
// 	"A Schur-Pade Algorithm for Fractional Powers of a Matrix."
// 	SIAM Journal on Matrix Analysis and Applications, 2011.
static complex float frac_power_superdiag(complex float l1, complex float l2, complex float t12, float p)
{
	if (l1 == l2)
		return t12 * p * cpowf(l1, p - 1);

	if (cabsf(l2 - l1) > cabsf(l1 + l2) / 2.)
		return t12 * (cpowf(l2, p) - cpowf(l1, p)) / (l2 - l1);

	complex float z = (l2 - l1) / (l2 + l1);

	int unwinding_num = (int)(ceilf((cimagf(clogf(l2) - clogf(l1)) - M_PI) / (2. * M_PI)));

	complex float tmp = 0.;

	if (unwinding_num)
		tmp = p * (catanhf(z) + M_PI * 1.i * unwinding_num);
	else
		tmp = p * catanhf(z);

	return t12 * cexpf(p / 2. * (clogf(l2) + clogf(l1))) * 2. * csinhf(tmp) / (l2 - l1);
}

// Superdiagonal entry of matrix logarithm
// 	N. J. Higham,
// 	"Functions of Matrices: Theory and Computation",
//	2011.
static complex float logm_superdiag(complex float l1, complex float l2, complex float t12)
{
	if (l1 == l2)
		return t12 / l1;

	if (cabsf(l2 - l1) > cabsf(l1 + l2) / 2.)
		return t12 * (clogf(l2) - clogf(l1)) / (l2 - l1);


	complex float z = (l2 - l1) / (l2 + l1);

	int unwinding_num = (int)(ceilf((cimagf(clogf(l2) - clogf(l1)) - M_PI) / (2. * M_PI)));

	complex float out = 0.;

	if (unwinding_num)
		out = 2. * t12 * (catanhf(z) + M_PI * 1.i * unwinding_num) / (l2 - l1);
	else
		out =  2. * t12 * catanhf(z) / (l2 - l1);

	return out;
}

// Matrix logarithm of upper triangular matrix
//	A. H. Al-Mohy and N. J. Higham
//	"Improved Inverse Scaling and Squaring Algorithms for the Matrix Logarithm"
//	SIAM Journal on Scientific Computing, 2012.
void logm_tri_matrix(int N, complex float out[N][N], const complex float in[N][N])
{
	// Table 2.1 in Al-Mohy and Higham, SIAM J. Sci. Comp., 2012.
	float theta[16] = {	1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2,
				1.28e-1, 2.06e-1, 2.88e-1, 3.67e-1,
				4.39e-1, 5.03e-1, 5.60e-1, 6.09e-1,
				6.52e-1, 6.89e-1, 7.21e-1, 7.49e-1 };

	complex float T[N][N];

	for (int i = 0; i < N; i++) {

		assert(0 != in[i][i]); // Otherwise find s will not terminate!

		for (int j = 0; j < N; j++) {

			T[i][j] = in[i][j];

			out[i][j] = 0.;
		}
	}
	// Find smallest s that fulfills a highest spectral radius of theta[6]

	complex float T_diag[N][N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			T_diag[i][j] = (i == j) ? T[i][j] : 0.;

	int s0 = 0;

	while (theta[6] < max_abs_diag(N, T_diag)) {

		matf_sqrt(N, N, T_diag, T_diag);
		s0++;
	}

	// Matrix square root

	complex float T_tmp[N][N];

	for (int i = 0; i < s0; i++) {

		sqrtm_tri_matrix(N, 32, T_tmp, T);
		mat_copy(N, N, T, T_tmp);
	}

	// Implementation of algorithm 4.1 in Al-Mohy and Higham, SIAM J. Sci. Comp., 2012.

	int s = s0;
	int k = 0;
	float d2 = powf(mat_onenorm_power(N, 2, T), 1. / 2.);
	float d3 = powf(mat_onenorm_power(N, 3, T), 1. / 3.);
	float a2 = (d3 > d2) ? d3 : d2;

	int pade_approx_deg = -1;

	if (theta[0] >= a2)
		pade_approx_deg = 0;
	else if (theta[1] >= a2)
		pade_approx_deg = 1;

	while (-1 == pade_approx_deg) {

		if (s0 < s)
			d3 = powf(mat_onenorm_power(N, 3, T), 1. / 3.);

		float d4 = powf(mat_onenorm_power(N, 4, T), 1. / 4.);

		float a3 = (d3 > d4) ? d3 : d4;

		if (theta[6] >= a3) {

			int j1 = 2000; // Random large initialization

			for (int i = 2; i <= 6; i++)
				j1 = ((theta[i] >= a3) && (j1 > i)) ? i : j1;

			if (5 >= j1) {

				pade_approx_deg = j1;
				break;
			}

			if ((theta[4] >= a3 / 2.) && (2 > k)) {

				sqrtm_tri_matrix(N, 32, T_tmp, T);
				mat_copy(N, N, T, T_tmp);

				k++;
				s++;
				continue;
			}
		}

		float d5 = powf(mat_onenorm_power(N, 5, T), 1. / 5.);

		float a4 = (d4 > d5) ? d4 : d5;
		float eta = (a3 < a4) ? a3 : a4;

		if (theta[5] >= eta) {

			pade_approx_deg = 5;
			break;
		}

		if (theta[6] >= eta) {

			pade_approx_deg = 6;
			break;
		}

		if (-1 != pade_approx_deg)
			break;

		sqrtm_tri_matrix(N, 32, T_tmp, T);
		mat_copy(N, N, T, T_tmp);

		s++;
	}

	pade_approx_deg++; // zero indexing vs ones-indexing in paper

	complex float R[N][N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			R[i][j] = T[i][j] - ((i == j) ? 1. : 0.);

	for (int i = 0; i < N; i++)
		R[i][i] = mat_briggs(s, in[i][i]);

	// Replace diagonal and first super-diagonal
	for (int i = 0; i < N-1; i++)
		R[i][i+1] = frac_power_superdiag(in[i][i], in[i+1][i+1], in[i][i+1], pow(2, -s));


	// U = 2^{s} * r_m(T - I) with partial fraction expansion

	double roots[pade_approx_deg];
	double weights[pade_approx_deg];

	roots_weights_gauss_legendre(pade_approx_deg, 2., roots, weights);

	// [-1, 1] -> [0, 1]
	double r_shift[pade_approx_deg];
	double w_shift[pade_approx_deg];

	for (int i = 0; i < pade_approx_deg; i++) {

		r_shift[i] = 0.5 + 0.5 * roots[i];
		w_shift[i] = 0.5 * weights[i];
	}

	for (int d = 0; d < pade_approx_deg; d++) {

		float ir = (float)r_shift[d];
		float iw = (float)w_shift[d];

		complex float M1[N][N];
		complex float M2[N][N];

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {

				M1[i][j] = ((i == j) ? 1. : 0.) + ir * R[i][j];
				M2[i][j] = iw * R[i][j];
			}
		}

		solve_tri_matrix(N, N, M1, M2, true);

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				out[i][j] += M2[i][j];
	}

	// loop could be combined, but might be easier to understand in this form
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			out[i][j] *= pow(2., s);

	// Recompute diagonal (FIXME: Add skipping option if principle branch exists?)
	for (int i = 0; i < N; i++)
		out[i][i] = clogf(in[i][i]);

	// Recompute superdiagonal
	for (int i = 0; i < N-1; i++)
		out[i][i+1] = logm_superdiag(in[i][i], in[i+1][i+1], in[i][i+1]);
}


// Input matrix is destroyed by schur decomposition
void mat_logm(int N, complex float out[N][N], complex float in[N][N])
{
	// Schur decomposition to transform input into upper triangular shape T
	complex float T[N][N];
	complex float Z[N][N];

	mat_schur(N, T, Z, in);

	// logm of upper triangular matrix
	complex float U[N][N];
	logm_tri_matrix(N, U, T);

	// Transform back from triangular shape
	mat_schur_recov(N, out, U, Z);
}
#endif

void unpack_tri_matrix(int N, complex float m[N][N], const complex float cov[N * (N + 1) / 2])
{
	int l = 0;

	for (int i = 0; i < N; i++)
		for (int j = 0; j <= i; j++)
			m[i][j] = cov[l++];
}

void gram_matrix2(int N, complex float cov[N * (N + 1) / 2], int L, const complex float data[N][L])
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

void matf_transpose(int A, int B, float dst[B][A], const float src[A][B])
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


void matf_copy(int N, int M, float out[N][M], const float in[N][M])
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			out[i][j] = in[i][j];
}

void (mat_conj)(int A, int B, complex float dst[A][B], const complex float src[A][B])
{
	for (int i = 0; i < A; i++)
		for (int j = 0; j < B; j++)
			dst[i][j] = conj(src[i][j]);
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


// Recover real symmetric matrix from band representation
void mat_band_reorder(int A, int B, double mat[A][A], double band[B][A], bool upper)
{
	int u = B - 1;

	for (int i = 0; i < A; i++) {
		for (int j = 0; j < A; j++) {

			// Restore upper or lower triangular matrix from band matrix
			if (u < abs(i - j)) {

				mat[i][j] = 0.;

			} else {

				if (upper) // Enforce symmetry
					mat[i][j] = (i <= j) ? band[u + i - j][j] : band[u + j - i][i];
				else // lower
					mat[i][j] = (i >= j) ? band[i - j][j] : band[j - i][i];
			}
		}
	}
}


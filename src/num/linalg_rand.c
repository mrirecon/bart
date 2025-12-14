/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <float.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/types.h"

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/lapack.h"

#include "linalg_rand.h"

#ifdef NO_LAPACK
void lapack_svd_econ(long M, long N, complex float U[(N > M) ? M : N][M], complex float VH[N][(N > M) ? M : N],
			float S[(N > M) ? M : N], complex float A[N][M]) { assert(0); }
void lapack_qr_econ(long M, long N,  complex float R[N][(N > M) ? M : N], complex float A[N][M]) { assert(0); }
void lapack_eig(long N, float eigenval[N], complex float matrix[N][N]) { assert(0); }
#endif

/*
 * Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions
 * N. Halko, P. G. Martinsson, and J. A. Tropp
 * SIAM Review 2011 53:2, 217-288
 *
 * We consider the linear operator to represent the matrix A : C^N -> C^M
 *
 * We denote the rank or blocksize to be K
 */

static void assert_linop(const struct operator_s* A, const struct operator_s* AH)
{
	const struct iovec_s* dom = operator_domain(A);
	const struct iovec_s* cod = operator_codomain(A);

	assert(iovec_check(operator_codomain(AH), dom->N, dom->dims, dom->strs));
	assert(iovec_check(operator_domain(AH), cod->N, cod->dims, cod->strs));
}

static long linop_get_columns(const struct operator_s* A, const struct operator_s* AH, long K)
{
	assert_linop(A, AH);

	const struct iovec_s* dom = operator_domain(A);

	if (1 == K)
		return md_calc_size(dom->N, dom->dims);

	int batch_dim = dom->N - 1;
	while (batch_dim > 0 && 1 == dom->dims[batch_dim])
		batch_dim--;

	assert(K == dom->dims[batch_dim]);
	return md_calc_size(batch_dim, dom->dims);
}

static long linop_get_rows(const struct operator_s* A, const struct operator_s* AH, long K)
{
	assert_linop(A, AH);

	const struct iovec_s* cod = operator_codomain(A);

	if (1 == K)
		return md_calc_size(cod->N, cod->dims);

	int batch_dim = cod->N - 1;
	while (batch_dim > 0 && 1 == cod->dims[batch_dim])
		batch_dim--;

	assert(K == cod->dims[batch_dim]);
	return md_calc_size(batch_dim, cod->dims);
}

static long linop_get_blocksize(const struct operator_s* A, const struct operator_s* AH)
{
	assert_linop(A, AH);

	const struct iovec_s* cod = operator_codomain(A);

	int batch_dim = cod->N - 1;
	while (batch_dim > 0 && 1 == cod->dims[batch_dim])
		batch_dim--;

	return cod->dims[batch_dim];
}


// algorithm 4.4
void randomized_subspace_iteration_block(const struct operator_s* A, const struct operator_s* AH, int q, long M, long K, complex float Q[K][M])
{
	assert(K == linop_get_blocksize(A, AH));
	assert(M == linop_get_rows(A, AH, K));
	long N = linop_get_columns(A, AH, K);

	complex float* Qptr = &Q[0][0];
	long cdims[2] = { M, K };
	long ddims[2] = { N, K };

	assert(K <= N);
	assert(K <= M);

	complex float* tmp = md_alloc_sameplace(2, ddims, CFL_SIZE, Qptr);
	md_gaussian_rand(2, ddims, tmp);

	operator_apply_unchecked(A, Qptr, tmp);
	lapack_qr_econ(M, K, NULL, MD_CAST_ARRAY2(complex float, 2, cdims, Qptr, 0, 1));

	for (int i = 0; i < q; i++) {

		operator_apply_unchecked(AH, tmp, Qptr);
		lapack_qr_econ(N, K, NULL, MD_CAST_ARRAY2(complex float, 2, ddims, tmp, 0, 1));

		operator_apply_unchecked(A, Qptr, tmp);
		lapack_qr_econ(M, K, NULL, MD_CAST_ARRAY2(complex float, 2, cdims, Qptr, 0, 1));
	}

	md_free(tmp);
}


// algorithm 5.1
void randomized_svd_block(const struct operator_s* A, const struct operator_s* AH, int q, long M, long N, long K, long P,
		     complex float U[K][M],
		     complex float VH[N][K],
		     float S[K])
{
	long KP = K + P;
	assert(KP == linop_get_blocksize(A, AH));
	assert(M == linop_get_rows(A, AH, KP));
	assert(N == linop_get_columns(A, AH, KP));

	complex float (*Q)[KP][M] = md_alloc(2, MD_DIMS(M, KP), CFL_SIZE);
	randomized_subspace_iteration_block(A, AH, q, M, KP, *Q);

	complex float (*BH)[KP][N] = md_alloc(2, MD_DIMS(N, KP), CFL_SIZE);
	operator_apply_unchecked(AH, &(*BH)[0][0], &(*Q)[0][0]);

	complex float (*B)[N][KP] = md_alloc(2, MD_DIMS(KP, N), CFL_SIZE);
	md_transpose(2, 0, 1, MD_DIMS(KP, N), &(*B)[0][0], MD_DIMS(N, KP), &(*BH)[0][0], CFL_SIZE);
	md_free(*BH);

	md_zconj(2, MD_DIMS(KP, N), &(*B)[0][0], &(*B)[0][0]);

	complex float (*Ut)[KP][KP] = (NULL != U) ? md_alloc(2, MD_DIMS(KP, KP), CFL_SIZE) : NULL;
	complex float (*VHt)[N][KP] = (NULL != U) ? md_alloc(2, MD_DIMS(KP, N), CFL_SIZE) : NULL;

	PTR_ALLOC(float[KP], S2);

	lapack_svd_econ(KP, N, (NULL != U) ? *Ut : NULL, (NULL != VH) ? *VHt : NULL, *S2, *B);
	for (int i = 0; i < K; i++)
		S[i] = (*S2)[i];

	PTR_FREE(S2);

	md_free(*B);

	if (NULL != U) {

		md_ztenmul(3, MD_DIMS(M, 1, K), &U[0][0], MD_DIMS(M, KP, 1), &(*Q)[0][0], MD_DIMS(1, KP, K), &(*Ut)[0][0]);
		md_free(*Ut);
	}

	md_free(*Q);

	if (NULL != VH) {

		md_copy_block(2, MD_DIMS(0, 0), MD_DIMS(K, N), &(VH)[0][0], MD_DIMS(KP, N), &(*VHt)[0][0], CFL_SIZE);
		md_free(*VHt);
	}


}

//FIXME: maybe we should use the cholesky decomposition as in algorithm 5.5
void randomized_eig_block(const struct operator_s* op, int q, long N, long K, long P, complex float U[K][N], float S[K])
{
	long KP = K + P;
	assert(KP == linop_get_blocksize(op, op));
	assert(N == linop_get_columns(op, op, KP));

	complex float (*Q)[KP][N] = md_alloc(2, MD_DIMS(N, KP), CFL_SIZE);
	randomized_subspace_iteration_block(op, op, q, N, KP, *Q);

	complex float (*B1)[KP][N] = md_alloc(2, MD_DIMS(N, KP), CFL_SIZE);
	operator_apply_unchecked(op, &(*B1)[0][0], &(*Q)[0][0]);

	complex float (*B2)[KP][KP] = md_alloc(2, MD_DIMS(KP, KP), CFL_SIZE);
	md_ztenmulc(3, MD_DIMS(1, KP, KP), &(*B2)[0][0], MD_DIMS(N, 1, KP), &(*B1)[0][0], MD_DIMS(N, KP, 1), &(*Q)[0][0]);

	PTR_ALLOC(float[KP], S2);

	lapack_eig(KP, *S2, *B2);
	for (int i = 0; i < K; i++)
		S[i] = (*S2)[i + P];

	PTR_FREE(S2);

	if (NULL != U)
		md_ztenmul(3, MD_DIMS(N, 1, K), &U[0][0], MD_DIMS(N, KP, 1), &(*Q)[0][0], MD_DIMS(1, KP, K), &(*B2)[P][0]);

	md_free(*B1);
	md_free(*B2);
	md_free(*Q);
}


struct matmul_s {

	operator_data_t super;

	long M;
	long N;
	long KP;

	const complex float* mat;
	bool adjoint;
};

static DEF_TYPEID(matmul_s);

static void matmul_apply(const operator_data_t* data, int N, void* args[N])
{
	const auto d = CAST_DOWN(matmul_s, data);
	assert(2 == N);

	if (d->adjoint)
		md_ztenmulc(3, MD_DIMS(1, d->N, d->KP), args[0], MD_DIMS(d->M, 1, d->KP), args[1], MD_DIMS(d->M, d->N, 1), d->mat);
	else
		md_ztenmul(3, MD_DIMS(d->M, 1, d->KP), args[0], MD_DIMS(1, d->N, d->KP), args[1], MD_DIMS(d->M, d->N, 1), d->mat);
}

static void matmul_free(const operator_data_t* data)
{
	const auto d = CAST_DOWN(matmul_s, data);
	xfree(d);
}

// this is a simple wrapper to be independent of the linop implementation in linops/fmac.c
static const struct operator_s* operator_matmul_create(long M, long N, long KP, const complex float mat[N][M], bool adjoint)
{
	PTR_ALLOC(struct matmul_s, data);
	SET_TYPEID(matmul_s, data);

	data->M = M;
	data->N = N;
	data->KP = KP;
	data->mat = mat[0];
	data->adjoint = adjoint;

	if (adjoint)
		return operator_create(2, MD_DIMS(N, KP), 2, MD_DIMS(M, KP), CAST_UP(PTR_PASS(data)), matmul_apply, matmul_free);
	else
		return operator_create(2, MD_DIMS(M, KP), 2, MD_DIMS(N, KP), CAST_UP(PTR_PASS(data)), matmul_apply, matmul_free);
}

void randomized_svd_dense(int q, long M, long N, long K, long P,
		     complex float U[K][M], complex float VH[N][K], float S[K],
		     const complex float mat[N][M])
{
	long KP = K + P;

	const struct operator_s* A = operator_matmul_create(M, N, KP, mat, false);
	const struct operator_s* AH = operator_matmul_create(M, N, KP, mat, true);

	randomized_svd_block(A, AH, q, M, N, K, P, U, VH, S);

	operator_free(A);
	operator_free(AH);
}

void randomized_eig_dense(int q, long N, long K, long P, complex float U[K][N], float S[K], const complex float mat[N][N])
{
	long KP = K + P;
	const struct operator_s* A = operator_matmul_create(N, N, KP, mat, false);

	randomized_eig_block(A, q, N, K, P, U, S);

	operator_free(A);
}


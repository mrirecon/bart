/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <math.h>
#include <complex.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/lapack.h"
#include "num/linalg_rand.h"

#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/fmac.h"

#include "utest.h"


static bool test_randomized_svd_diag(void)
{
	int N = 32;
	complex float diag[N];

	for (int i = 0; i < N; i++)
		diag[i] = i;

	int K = 12;
	float sing[K];

	const struct linop_s* op = linop_cdiag_create(2, MD_DIMS(N, K), MD_BIT(0), diag);
	randomized_svd_block(op->forward, op->adjoint, 10, N, N, K, 0, NULL, NULL, sing);
	linop_free(op);

	float err = 0.;
	float nrm = 0.;

	for (int i = 0; i < 4; i++) {

		err += powf(sing[i] - (N - 1 - i), 2.);
		nrm += powf((N - 1 - i), 2.);
	}

	err = sqrtf(err) / sqrtf(nrm);

	UT_RETURN_ASSERT_TOL(err, UT_TOL);
}


UT_REGISTER_TEST(test_randomized_svd_diag);


static bool test_randomized_eig_diag(void)
{
	int N = 32;
	complex float diag[N];

	for (int i = 0; i < N; i++)
		diag[i] = i;

	int K = 12;
	float sing[K];

	const struct linop_s* op = linop_cdiag_create(2, MD_DIMS(N, K), MD_BIT(0), diag);
	randomized_eig_block(op->normal, 10, N, K, 0, NULL, sing);
	linop_free(op);

	float err = 0.;
	float nrm = 0.;

	for (int i = 0; i < 4; i++) {

		err += powf(sing[K - 1 - i] - (N - 1 - i) * (N - 1 - i), 2.);
		nrm += powf((N - 1 - i) * (N - 1 - i), 2.);
	}

	err = sqrtf(err) / sqrtf(nrm);

	UT_RETURN_ASSERT_TOL(err, UT_TOL);
}


UT_REGISTER_TEST(test_randomized_eig_diag);


static void svd_lowrank(long M, long N, long K, complex float approx[N][M], const complex float imat[N][M], bool randomized, long p)
{
	long Adims[3] = { M, 1, N };
	md_copy(3, Adims, approx, imat, CFL_SIZE);

	long Udims[] = { M, M, 1 };
	long VHdims[] = { 1, N, N };
	long Sdims[] = { 1, N, 1 };

	if (randomized) {

		Udims[1] = K + p;
		VHdims[1] = K + p;
	}

	complex float* U = md_alloc(3, Udims, CFL_SIZE);
	complex float* VH = md_alloc(3, VHdims, CFL_SIZE);

	float rsigma[MAX(M, N)];

	if (randomized) {

		long Adims[4] = { M, 1, N, K + p };
		const struct linop_s* lop_fmac = linop_fmac_create(4, Adims, MD_BIT(2), MD_BIT(0), MD_BIT(3), imat[0]);

		randomized_svd_block(lop_fmac->forward, lop_fmac->adjoint, 10, M, N, K + p, 0,
					MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1),
					MD_CAST_ARRAY2(complex float, 3, VHdims, VH, 1, 2),
					rsigma);

		linop_free(lop_fmac);
	} else {

		lapack_svd(M, N, 	MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1),
					MD_CAST_ARRAY2(complex float, 3, VHdims, VH, 1, 2),
					rsigma,
					MD_CAST_ARRAY2(complex float, 3, Adims, approx, 0, 2));
	}

	complex float sigma[MAX(M, N)];
	memset(sigma, 0, sizeof(sigma));
	for (int i = 0; i < MIN(M, N); i++)
		sigma[i] = rsigma[i];

	md_zmul2(3, VHdims, MD_STRIDES(3, VHdims, CFL_SIZE), VH, MD_STRIDES(3, VHdims, CFL_SIZE), VH, MD_STRIDES(3, Sdims, CFL_SIZE), sigma);
	md_ztenmul2(3, MD_DIMS(M, K, N), MD_STRIDES(3, Adims, CFL_SIZE), approx[0], MD_STRIDES(3, Udims, CFL_SIZE), U, MD_STRIDES(3, VHdims, CFL_SIZE), VH);

	md_free(U);
	md_free(VH);
}


static bool test_randomized_svd_lowrank1(void)
{
	long M = 22;
	long N = 33;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	complex float* mat1 = md_alloc(3, Adims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, Adims, CFL_SIZE);

	svd_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	svd_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, Adims, mat1, mat2);

	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 10. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_svd_lowrank1);


static bool test_randomized_svd_lowrank2(void)
{
	long M = 33;
	long N = 22;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	complex float* mat1 = md_alloc(3, Adims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, Adims, CFL_SIZE);

	svd_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	svd_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, Adims, mat1, mat2);

	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 10. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_svd_lowrank2);


static void normal_lowrank(long M, long N, long K, complex float approx[N][N], const complex float imat[N][M], bool randomized, long p)
{
	long AHAdims[3] = { N, 1, N };

	long Udims[] = { N, N, 1 };
	long Sdims[] = { 1, N, 1 };

	if (randomized)
		Udims[1] = K + p;

	complex float* U = md_alloc(3, Udims, CFL_SIZE);
	float rsigma[N];
	memset(rsigma, 0, sizeof(rsigma));

	if (randomized) {

		long Adims[4] = { M, 1, N, K + p };
		const struct linop_s* lop_fmac = linop_fmac_create(4, Adims, MD_BIT(2), MD_BIT(0), MD_BIT(3), imat[0]);

		randomized_eig_block(lop_fmac->normal, 10, N, K + p, 0,
					MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1),
					rsigma);

		linop_free(lop_fmac);
	} else {
		complex float (*AHA)[N][N] = md_alloc(2, MD_DIMS(N, N), CFL_SIZE);
		md_ztenmulc(3, MD_DIMS(1, N, N), &(*AHA)[0][0], MD_DIMS(M, 1, N), (complex float*)imat, MD_DIMS(M, N, 1), (complex float*)imat);

		md_copy(3, AHAdims, U, &(*AHA)[0][0], CFL_SIZE);
		lapack_eig(N, rsigma, MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1));
		md_free(*AHA);
	}

	complex float sigma[N];
	memset(sigma, 0, sizeof(sigma));
	for (int i = 0; i < N; i++)
		sigma[i] = sqrtf(0 > rsigma[i] ? 0. : rsigma[i]);


	md_zmul2(3, Udims, MD_STRIDES(3, Udims, CFL_SIZE), U, MD_STRIDES(3, Udims, CFL_SIZE), U, MD_STRIDES(3, Sdims, CFL_SIZE), sigma);


	long UHdims[3] = { 1, Udims[1], N };
	complex float* UH = md_alloc(3, UHdims, CFL_SIZE);
	md_transpose(3, 0, 2, UHdims, UH, Udims, U, CFL_SIZE);
	md_zconj(3, UHdims, UH, UH);

	complex float* Uoff = &MD_ACCESS(3, MD_STRIDES(3, Udims, CFL_SIZE), MD_DIMS(0, Udims[1] - K, 0), U);
	complex float* UHoff = &MD_ACCESS(3, MD_STRIDES(3, UHdims, CFL_SIZE), MD_DIMS(0, Udims[1] - K, 0), UH);

	md_ztenmul2(3, MD_DIMS(N, K, N), MD_STRIDES(3, AHAdims, CFL_SIZE), approx[0], MD_STRIDES(3, Udims, CFL_SIZE), Uoff, MD_STRIDES(3, UHdims, CFL_SIZE), UHoff);

	md_free(U);
	md_free(UH);
}



static bool test_randomized_eig_lowrank1(void)
{
	long M = 22;
	long N = 33;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	long AHAdims[3] = { N, 1, N };

	complex float* mat1 = md_alloc(3, AHAdims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, AHAdims, CFL_SIZE);

	normal_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	normal_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, AHAdims, mat1, mat2);

	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 10. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_eig_lowrank1);


static bool test_randomized_eig_lowrank2(void)
{
	long M = 33;
	long N = 22;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	long AHAdims[3] = { N, 1, N };

	complex float* mat1 = md_alloc(3, AHAdims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, AHAdims, CFL_SIZE);

	normal_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	normal_lowrank(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, AHAdims, mat1, mat2);

	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 10. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_eig_lowrank2);


static void normal_lowrank_dense(long M, long N, long K, complex float approx[N][N], const complex float imat[N][M], bool randomized, long P)
{
	complex float (*AHA)[N][N] = md_alloc(2, MD_DIMS(N, N), CFL_SIZE);
	md_ztenmulc(3, MD_DIMS(1, N, N), &(*AHA)[0][0], MD_DIMS(M, 1, N), (complex float*)imat, MD_DIMS(M, N, 1), (complex float*)imat);

	long AHAdims[3] = { N, 1, N };

	long Udims[] = { N, N, 1 };
	long Sdims[] = { 1, N, 1 };

	if (randomized)
		Udims[1] = K;

	complex float* U = md_alloc(3, Udims, CFL_SIZE);
	float rsigma[N];
	memset(rsigma, 0, sizeof(rsigma));

	if (randomized) {

		randomized_eig_dense(10, N, K, P, MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1), rsigma, (*AHA));
	} else {

		md_copy(3, AHAdims, U, &(*AHA)[0][0], CFL_SIZE);
		lapack_eig(N, rsigma, MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1));
	}

	complex float sigma[N];
	memset(sigma, 0, sizeof(sigma));
	for (int i = 0; i < N; i++)
		sigma[i] = sqrtf(0 > rsigma[i] ? 0. : rsigma[i]);


	md_zmul2(3, Udims, MD_STRIDES(3, Udims, CFL_SIZE), U, MD_STRIDES(3, Udims, CFL_SIZE), U, MD_STRIDES(3, Sdims, CFL_SIZE), sigma);


	long UHdims[3] = { 1, Udims[1], N };
	complex float* UH = md_alloc(3, UHdims, CFL_SIZE);
	md_transpose(3, 0, 2, UHdims, UH, Udims, U, CFL_SIZE);
	md_zconj(3, UHdims, UH, UH);

	complex float* Uoff = &MD_ACCESS(3, MD_STRIDES(3, Udims, CFL_SIZE), MD_DIMS(0, Udims[1] - K, 0), U);
	complex float* UHoff = &MD_ACCESS(3, MD_STRIDES(3, UHdims, CFL_SIZE), MD_DIMS(0, Udims[1] - K, 0), UH);

	md_ztenmul2(3, MD_DIMS(N, K, N), MD_STRIDES(3, AHAdims, CFL_SIZE), approx[0], MD_STRIDES(3, Udims, CFL_SIZE), Uoff, MD_STRIDES(3, UHdims, CFL_SIZE), UHoff);

	md_free(U);
	md_free(UH);
	md_free(*AHA);
}

static bool test_randomized_eig_lowrank1_dense(void)
{
	long M = 22;
	long N = 33;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	long AHAdims[3] = { N, 1, N };

	complex float* mat1 = md_alloc(3, AHAdims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, AHAdims, CFL_SIZE);

	normal_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	normal_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, AHAdims, mat1, mat2);

	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 10. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_eig_lowrank1_dense);

#if 0
static void debug_matrix(int M, int N, const complex float mat[N][M])
{
	for (int i = 0; i < N; i++) {

		for (int j = 0; j < M; j++)
			debug_printf(DP_INFO, "%+.2f%+.2fi ", crealf(mat[i][j]), cimag(mat[i][j]));

		debug_printf(DP_INFO, "\n");
	}

	debug_printf(DP_INFO, "\n");
}
#endif


static bool test_randomized_eig_lowrank2_dense(void)
{
	long M = 14;
	long N = 14;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	long AHAdims[3] = { N, 1, N };

	complex float* mat1 = md_alloc(3, AHAdims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, AHAdims, CFL_SIZE);

	normal_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	normal_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, AHAdims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, AHAdims, mat1, mat2);


	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 10. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_eig_lowrank2_dense);


static void svd_lowrank_dense(long M, long N, long K, complex float approx[N][M], const complex float imat[N][M], bool randomized, long p)
{
	long Adims[3] = { M, 1, N };
	md_copy(3, Adims, approx, imat, CFL_SIZE);

	long Udims[] = { M, M, 1 };
	long VHdims[] = { 1, N, N };
	long Sdims[] = { 1, N, 1 };

	if (randomized) {

		Udims[1] = K;
		VHdims[1] = K;
	}

	complex float* U = md_alloc(3, Udims, CFL_SIZE);
	complex float* VH = md_alloc(3, VHdims, CFL_SIZE);

	float rsigma[MAX(M, N)];

	if (randomized) {

		randomized_svd_dense(10, M, N, K, p,
					MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1),
					MD_CAST_ARRAY2(complex float, 3, VHdims, VH, 1, 2),
					rsigma,
					MD_CAST_ARRAY2(complex float, 3, Adims, approx, 0, 2));
	} else {

		lapack_svd(M, N, 	MD_CAST_ARRAY2(complex float, 3, Udims, U, 0, 1),
					MD_CAST_ARRAY2(complex float, 3, VHdims, VH, 1, 2),
					rsigma,
					MD_CAST_ARRAY2(complex float, 3, Adims, approx, 0, 2));
	}

	complex float sigma[MAX(M, N)];
	memset(sigma, 0, sizeof(sigma));
	for (int i = 0; i < MIN(M, N); i++)
		sigma[i] = rsigma[i];

	md_zmul2(3, VHdims, MD_STRIDES(3, VHdims, CFL_SIZE), VH, MD_STRIDES(3, VHdims, CFL_SIZE), VH, MD_STRIDES(3, Sdims, CFL_SIZE), sigma);
	md_ztenmul2(3, MD_DIMS(M, K, N), MD_STRIDES(3, Adims, CFL_SIZE), approx[0], MD_STRIDES(3, Udims, CFL_SIZE), U, MD_STRIDES(3, VHdims, CFL_SIZE), VH);

	md_free(U);
	md_free(VH);
}


static bool test_randomized_svd_lowrank1_dense(void)
{
	long M = 22;
	long N = 33;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	complex float* mat1 = md_alloc(3, Adims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, Adims, CFL_SIZE);

	svd_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	svd_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, Adims, mat1, mat2);

	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 10. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_svd_lowrank1_dense);


static bool test_randomized_svd_lowrank2_dense(void)
{
	long M = 33;
	long N = 22;

	long Adims[3] = { M, 1, N };

	complex float* init = md_alloc(3, Adims, CFL_SIZE);
	md_gaussian_rand(3, Adims, init);

	complex float* mat1 = md_alloc(3, Adims, CFL_SIZE);
	complex float* mat2 = md_alloc(3, Adims, CFL_SIZE);

	svd_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat1, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), false, 10);
	svd_lowrank_dense(M, N, 4, MD_CAST_ARRAY2(complex float, 3, Adims, mat2, 0, 2), MD_CAST_ARRAY2(complex float, 3, Adims, init, 0, 2), true, 10);
	float err = md_znrmse(3, Adims, mat1, mat2);

	md_free(init);
	md_free(mat1);
	md_free(mat2);

	UT_RETURN_ASSERT_TOL(err, 50. * UT_TOL);
}

UT_REGISTER_TEST(test_randomized_svd_lowrank2_dense);


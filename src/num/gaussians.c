/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <math.h>
#include <complex.h>
#include <assert.h>

#include "num/flpmath.h"
#include "num/rand.h"
#include "num/linalg.h"
#include "num/multind.h"

#include "gaussians.h"


float gaussian_pdf(int N, const complex float m[N], const
		complex float icov[N][N], const complex float x[N])
{
	complex float u[N];

	vec_copy(N, u, x);
	vec_saxpy(N, u, -1., m);

	complex float v[N];
	mat_vecmul(N, N, v, icov, u);

	float sum = crealf(vec_dot(N, v, u));

	float f = powf(M_PI, -N);
	float d = cabsf(mat_det(N, icov));

	return f * d * expf(-sum);
}


float gaussian_mix_pdf(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float icov[M][N][N], const complex float x[N])
{
	float sum = 0.;

	for (int i = 0; i < M; i++)
		sum += coeff[i];

	assert(fabsf(1. - sum) < 1.E-6);

	sum = 0.;
	for (int i = 0; i < M; i++)
		sum += coeff[i] * gaussian_pdf(N, m[i], icov[i], x);

	return sum;
}



void gaussian_sample(int N, const complex float m[N],
		const complex float icov[N][N], complex float x[N])
{
	complex float u[N] = { }; // maybe-uninitialized

	for (int i = 0; i < N; i++)
		u[i] = gaussian_rand() / sqrtf(2.);

	complex float sqr_cov[N][N];
	mat_inverse(N, sqr_cov, icov);
#if 0
	for (int i = 0; i < N; i++)
		sqr_cov[i][i] += 0.001;	// epislon
#endif
	cholesky(N, sqr_cov);

	mat_vecmul(N, N, x, sqr_cov, u);

	for (int i = 0; i < N; i++)
		x[i] += m[i];
}


void gaussian_mix_sample(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float icov[M][N][N], complex float x[N])
{
	float r = uniform_rand();

	int ind;
	float icoeff = 0.;
	for (ind = 0; ind < M; ind++) {

		icoeff += coeff[ind];

		if (r <= icoeff)
			break;
	}

	assert(1.0001 >= icoeff);
	assert(ind <= M);
	if (ind == M)
		ind--;

	gaussian_sample(N, m[ind], icov[ind], x);
}



void gaussian_score(int N, const complex float m[N], const complex float icov[N][N],
		const complex float x[N], complex float sc[N])
{
	complex float u[N];
	vec_zero(N, u);
	vec_saxpy(N, u, -2., x);
	vec_saxpy(N, u, +2., m);
	mat_vecmul(N, N, sc, icov, u);
}


void gaussian_mix_score(int M, int N, const float coeff[M], const complex float m[M][N],
		const complex float icov[M][N][N],
		const complex float x[N], complex float sc[N])
{
	vec_zero(N, sc);
	float c[M];
	float no = 0.;

	for (int i = 0; i < M; i++) {

		c[i] = coeff[i] * gaussian_pdf(N, m[i], icov[i], x);
		no += c[i];
	}

	for (int i = 0; i < M; i++) {

		complex float u[N];
		gaussian_score(N, m[i], icov[i], x, u);

		vec_saxpy(N, sc, c[i] / no, u);
	}
}


void gaussian_convolve(int N, complex float m[N],
		complex float icov[N][N],
		const complex float m1[N],
		const complex float icov1[N][N],
		const complex float m2[N],
		const complex float icov2[N][N])
{
	vec_copy(N, m, m1);
	vec_saxpy(N, m, 1., m2);

	complex float cov1[N][N];
	mat_inverse(N, cov1, icov1);

	complex float cov2[N][N];
	mat_inverse(N, cov2, icov2);

	complex float cov[N][N];
	mat_add(N, N, cov, cov1, cov2);
	mat_inverse(N, icov, cov);
}


void gaussian_multiply(int N, complex float m[N],
		complex float icov[N][N],
		const complex float m1[N],
		const complex float icov1[N][N],
		const complex float m2[N],
		const complex float icov2[N][N])
{
	complex float m1b[N];
	mat_vecmul(N, N, m1b, icov1, m1);

	complex float m2b[N];
	mat_vecmul(N, N,  m2b, icov2, m2);

	vec_saxpy(N, m1b, 1., m2b);

	mat_add(N, N, icov, icov1, icov2);

	complex float cov[N][N];
	mat_inverse(N, cov, icov);
	mat_vecmul(N, N, m, cov, m1b);
}


float gaussian_multiply_factor(int N,
		const complex float m1[N],
		const complex float icov1[N][N],
		const complex float m2[N],
		const complex float icov2[N][N])
{
	complex float cov1[N][N];
	mat_inverse(N, cov1, icov1);

	complex float cov2[N][N];
	mat_inverse(N, cov2, icov2);

	complex float cov[N][N];
	mat_add(N, N, cov, cov1, cov2);

	complex float icov[N][N];
	mat_inverse(N, icov, cov);

	return gaussian_pdf(N, m1, icov, m2);
}


/**
 * Calculates the gradient of the Gaussian exponent
 *
 * grad = 2 * (x - mus) / vars; dims -> {n,m,C,B}
 *
 * where:	n, m = dimension of sample
 *		C = number of Gaussians
 *		B = batchsize (i.e. number of samples)
 **/
void md_grad_gaussian(int D, const long dims_grad[D],
	complex float* grad, const long dims_x[D], const complex float* x, const long dims_mu[D],
	const complex float* mu, const long dims_vars[D], const complex float* vars)
{
	long strs_x[D];
	long strs_mu[D];
	long strs_vars[D];
	long strs_grad[D];

	md_calc_strides(D, strs_x, dims_x, CFL_SIZE);
	md_calc_strides(D, strs_mu, dims_mu, CFL_SIZE);
	md_calc_strides(D, strs_vars, dims_vars, CFL_SIZE);
	md_calc_strides(D, strs_grad, dims_grad, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(D, dims_grad, CFL_SIZE, x);

	//FIXME: workaraound as efficient strides for sub and div not available
	complex float* tmu = md_alloc_sameplace(D, dims_mu, CFL_SIZE, mu);
	complex float* tvars = md_alloc_sameplace(D, dims_vars, CFL_SIZE, vars);

	md_zsmul(D, dims_mu, tmu, mu, -1);
	md_zspow(D, dims_vars, tvars, vars, -1);

	md_zadd2(D, dims_grad, strs_grad, tmp, strs_x, x, strs_mu, tmu);
	md_zmul2(D, dims_grad, strs_grad, grad, strs_grad, tmp, strs_vars, tvars);
	md_zsmul2(D, dims_grad, strs_grad, grad, strs_grad, grad, 2);

	md_free(tmu);
	md_free(tvars);
	md_free(tmp);
}

/**
 * Calculates the log-expression of given Gaussians
 *
 * log_gauss = - 0.5 * n * m * log(pi) - 0.5 * n * m * log(vars) - (x - mus) / vars @ (x - mus); dims -> {1,1,C,B}
 *
 **/
void md_log_gaussian(int D, const long dims_log_gauss[D], complex float* log_gauss,
		const long dims_x[D], const complex float* x,
		const long dims_mu[D], const complex float* mu,
		const long dims_vars[D], const complex float* vars)
{
	long dims_grad[D];
	md_max_dims(D, ~0ul, dims_grad, dims_mu, dims_x);

	long strs_x[D];
	long strs_mu[D];
	long strs_vars[D];
	long strs_log_gauss[D];
	long strs_grad[D];

	md_calc_strides(D, strs_x, dims_x, CFL_SIZE);
	md_calc_strides(D, strs_mu, dims_mu, CFL_SIZE);
	md_calc_strides(D, strs_vars, dims_vars, CFL_SIZE);
	md_calc_strides(D, strs_log_gauss, dims_log_gauss, CFL_SIZE);
	md_calc_strides(D, strs_grad, dims_grad, CFL_SIZE);

	complex float* grad = md_alloc_sameplace(D, dims_grad, CFL_SIZE, x);
	complex float* diff = md_alloc_sameplace(D, dims_grad, CFL_SIZE, x);
	complex float* xmuvarxmu = md_alloc_sameplace(D, dims_grad, CFL_SIZE, x);
	complex float* log_vars = md_alloc_sameplace(D, dims_vars, CFL_SIZE, x);
	complex float* tmp0 = md_alloc_sameplace(D, dims_vars, CFL_SIZE, x);

	md_grad_gaussian(D, dims_grad, grad, dims_x, x, dims_mu, mu, dims_vars, vars);

	//FIXME: workaraound as efficient strides for sub not available
	complex float* tmu = md_alloc_sameplace(D, dims_mu, CFL_SIZE, mu);
	md_zsmul(D, dims_mu, tmu, mu, -1);
	md_zadd2(D, dims_grad, strs_grad, diff, strs_x, x, strs_mu, tmu);
	//md_zsub2(D, dims_grad, strs_grad, diff, strs_x, x, strs_mu, mu);

	md_ztenmulc2(D, dims_grad, strs_log_gauss, xmuvarxmu, strs_grad, diff, strs_grad, grad);
	md_zlog2(D, dims_vars, strs_vars, log_vars, strs_vars, vars);
	md_zsmul2(D, dims_vars, strs_vars, tmp0, strs_vars, log_vars, -0.5 * dims_x[0] * dims_x[1]);

	md_zsadd2(D, dims_vars, strs_vars, tmp0, strs_vars, tmp0, -0.5 * dims_x[0] * dims_x[1] * M_PI);

	md_zsmul2(D, dims_log_gauss, strs_log_gauss, log_gauss, strs_log_gauss, xmuvarxmu, -1);
	md_zaxpy2(D, dims_log_gauss, strs_log_gauss, log_gauss, 1, strs_vars, tmp0);

	md_free(grad);
	md_free(diff);
	md_free(xmuvarxmu);
	md_free(log_vars);
	md_free(tmp0);
	md_free(tmu);
}


/**
 * Calculates the weigthing of the Gaussians according to the log-sum-exp trick (https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
 *
 * gamma = exp(z - zmax) / sum_C exp(z - zmax); dims -> {1,1,C,B}
 *
 * where:	z = log(ws) - 0.5 * n * m * log(pi) - 0.5 * n * m * log(vars) - (x - mus) / vars @ (x - mus); dims -> {1,1,C,B}
 *		zmax =  max_C (z); dims -> {1,1,1,B}
 *
 **/
void md_mixture_weights(int D, const long dims_gamma[D], complex float* gamma,
		const long dims_log_gauss[D], complex float* log_gauss,
		const long dims_ws[D], const complex float* ws)
{
	long dims_zmax[D];
	unsigned long flags = 0; // flags for batchsize and number of gaussians

	flags = md_nontriv_dims(D, dims_log_gauss) & ~md_nontriv_dims(D, dims_ws);

	md_select_dims(D, flags, dims_zmax, dims_gamma);

	long dims_nb_gauss[D];
	md_min_dims(D, ~0u, dims_nb_gauss, dims_ws, dims_gamma); // zmax has batdim

	long strs_log_gauss[D];
	long strs_ws[D];
	long strs_gamma[D];
	long strs_zmax[D];

	md_calc_strides(D, strs_log_gauss, dims_log_gauss, CFL_SIZE);
	md_calc_strides(D, strs_ws, dims_ws, CFL_SIZE);
	md_calc_strides(D, strs_gamma, dims_gamma, CFL_SIZE);
	md_calc_strides(D, strs_zmax, dims_zmax, CFL_SIZE);

	complex float* z = md_alloc_sameplace(D, dims_log_gauss, CFL_SIZE, log_gauss);
	complex float* zmax = md_alloc_sameplace(D, dims_zmax, CFL_SIZE, log_gauss);
	complex float* tmp = md_alloc_sameplace(D, dims_log_gauss, CFL_SIZE, log_gauss);
	complex float* tmp1 = md_alloc_sameplace(D, dims_zmax, CFL_SIZE, log_gauss);
	complex float* tmp2 = md_alloc_sameplace(D, dims_log_gauss, CFL_SIZE, log_gauss);

	md_zlog2(D, dims_log_gauss, strs_log_gauss, z, strs_ws, ws);
	md_zaxpy2(D, dims_log_gauss, strs_log_gauss, z, 1, strs_log_gauss, log_gauss);
	md_reduce_zmax(D, dims_log_gauss, md_nontriv_dims(D, dims_nb_gauss), zmax, z);

	//FIXME: workaraound as efficient strides for sub not available
	complex float* tzmax = md_alloc_sameplace(D, dims_zmax, CFL_SIZE, zmax);

	md_zsmul(D, dims_zmax, tzmax, zmax, -1.);
	md_zadd2(D, dims_log_gauss, strs_log_gauss, tmp, strs_log_gauss, z, strs_zmax, tzmax);
	//md_zsub2(D, dims_log_gauss, strs_log_gauss, tmp, strs_log_gauss, z, strs_zmax, zmax);

	md_zexp2(D, dims_log_gauss, strs_log_gauss, tmp2, strs_log_gauss, tmp);
	md_zsum(D, dims_log_gauss, md_nontriv_dims(D, dims_nb_gauss), tmp1, tmp2);

	//FIXME: workaraound as efficient strides for div not available
	complex float* tmp3 = md_alloc_sameplace(D, dims_zmax, CFL_SIZE, zmax);

	md_zspow(D, dims_zmax, tmp3, tmp1, -1.);
	md_zmul2(D,  dims_gamma, strs_gamma, gamma, strs_log_gauss, tmp2, strs_zmax, tmp3);
	//md_zdiv2(D, dims_gamma, strs_gamma, gamma, strs_log_gauss, tmp2, strs_zmax, tmp1);

	md_free(z);
	md_free(zmax);
	md_free(tmp);
	md_free(tmp1);
	md_free(tmp2);
	md_free(tmp3);
	md_free(tzmax);
}

/**
 * Calculates the score of the GMM according to the log-sum-exp trick
 *
 * score = - sum_C gamma * 2 * (x - mus) / vars; dims -> {n,m,1,B}
 *
 * where:	C = number of Gaussians in the GMM
 *		gamma = weigthing term of the log-sum-exp trick; dims -> {1,1,C,B}
 *		x = input variable; dims -> {n,m,1,B}
 *		mus = mean of the Gaussians; dims -> {n,m,C,1}
 *
 **/
void md_gaussian_score(int D, const long dims_score[D], complex float* score,
		const long dims_x[D], const complex float* x,
		const long dims_mu[D], const complex float* mu,
		const long dims_vars[D], const complex float* vars,
		const long dims_ws[D], const complex float* ws)
{
	long grad_dims[D];
	md_max_dims(D, ~0ul, grad_dims, dims_mu, dims_score);

	assert(md_check_compat(D, ~0UL, grad_dims, dims_x));
	assert(md_check_compat(D, ~0UL, grad_dims, dims_mu));
	assert(md_check_equal_dims(D, dims_score, dims_x, ~0UL));

	long dims_log_gauss[D];
	unsigned long flags = 0; // flags for batchsize and number of gaussians
	flags = md_nontriv_dims(D, dims_mu) & md_nontriv_dims(D, dims_x);
	md_select_dims(D, ~flags, dims_log_gauss, grad_dims);

	long nb_gauss_dims[D];
	md_min_dims(D, ~0u, nb_gauss_dims, dims_vars, dims_mu);

	long strs_grad[D];
	md_calc_strides(D, strs_grad, grad_dims, CFL_SIZE);

	complex float* log_gauss = md_alloc_sameplace(D, dims_log_gauss, CFL_SIZE, x);
	complex float* gamma = md_alloc_sameplace(D, dims_log_gauss, CFL_SIZE, x);
	complex float* grad = md_alloc_sameplace(D, grad_dims, CFL_SIZE, x);

	md_log_gaussian(D, dims_log_gauss, log_gauss, dims_x, x, dims_mu, mu, dims_vars, vars);
	md_mixture_weights(D, dims_log_gauss, gamma, dims_log_gauss, log_gauss, dims_ws, ws);
	md_grad_gaussian(D, grad_dims, grad, dims_x, x, dims_mu, mu, dims_vars, vars);
	md_zmul2(D, grad_dims, strs_grad, grad, MD_STRIDES(D, dims_log_gauss, CFL_SIZE), gamma, strs_grad, grad);
	md_zsmul2(D, grad_dims, strs_grad, grad, strs_grad, grad, -1);

	md_zsum(D, grad_dims, ~md_nontriv_dims(D, dims_score), score, grad);

	md_free(log_gauss);
	md_free(gamma);
	md_free(grad);
}


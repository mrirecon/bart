/* Copyright 2013-2017. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Center Göttingen.
 * Copyright 2023-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2024 Martin Uecker <uecker@tugraz.at>
 * 2013-2014 Frank Ong <frankong@berkeley.edu>
 * 2013-2014,2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 *
 * Landweber L. An iteration formula for Fredholm integral equations of the
 * first kind. Amer. J. Math. 1951; 73, 615-624.
 *
 * Nesterov Y. A method of solving a convex programming problem with
 * convergence rate O (1/k2). Soviet Mathematics Doklady 1983; 27(2):372-376
 *
 * Bakushinsky AB. Iterative methods for nonlinear operator equations without
 * regularity. New approach. In Dokl. Russian Acad. Sci 1993; 330:282-284.
 *
 * Daubechies I, Defrise M, De Mol C. An iterative thresholding algorithm for
 * linear inverse problems with a sparsity constraint.
 * Comm Pure Appl Math 2004; 57:1413-1457.
 *
 * Beck A, Teboulle M. A fast iterative shrinkage-thresholding algorithm for
 * linear inverse problems. SIAM Journal on Imaging Sciences 2.1 2009; 183-202.
 *
 * Chambolle A, Pock, T. A First-Order Primal-Dual Algorithm for Convex Problems
 * with Applications to Imaging. J. Math. Imaging Vis. 2011; 40, 120-145.
 * 
 * Bredies K, Holler M. A TGV-Based Framework for Variational Image Decompression, 
 * Zooming, and Reconstruction. Part II: Numerics. SIAM J. Imaging Sci. 2015; 8, 2851-2886.
 */

#include <math.h>
#include <stdbool.h>

#include <stdio.h>
#include <string.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "iter/vec.h"
#include "iter/monitor.h"
#include "iter/monitor_iter6.h"

// FIXME: shouldn't this be a monitor?
#include "iter/iter_dump.h"

#include "italgos.h"
#include "misc/types.h"

extern inline void iter_op_call(struct iter_op_s op, float* dst, const float* src);
extern inline void iter_nlop_call(struct iter_nlop_s op, int N, float* args[N]);
extern inline void iter_nlop_call_select_der(struct iter_nlop_s op, int N, float* args[N], unsigned long der_out, unsigned long der_in);
extern inline void iter_op_p_call(struct iter_op_p_s op, float rho, float* dst, const float* src);
extern inline void iter_op_arr_call(struct iter_op_arr_s op, int NO, unsigned long oflags, float* dst[NO], int NI, unsigned long iflags, const float* src[NI]);



/* Liang J., Luo T, Schönlieb, C-B.
 * Improving “Fast Iterative Shrinkage-Thresholding Algorithm”: Faster, Smarter, and Greedier,
 * SIAM Journal on Scientific Computing 2022;44:A1069-A1091.
 *
 * (via. Lee et al. MRM 2024;91:1464-1477.)
 */

struct ravine_conf ravine_mod = { 1.f / 20.f, 1.f / 2.f, 4.f };
struct ravine_conf ravine_classical = { 1.f, 1.f, 4.f };

static float fista_formula(const struct ravine_conf conf, float ft)
{
	return (conf.p + sqrtf(conf.q + conf.r * ft * ft)) / 2.f;
}

/**
 * ravine step
 * (Nesterov 1983)
 */
static void ravine(const struct vec_iter_s* vops, const struct ravine_conf conf,
		long N, float* ftp, float* xa, float* xb)
{
	float ft = *ftp;
	float tfo = ft;

	ft = fista_formula(conf, ft);
	*ftp = ft;

	vops->swap(N, xa, xb);
	vops->axpy(N, xa, (1.f - tfo) / ft - 1.f, xa);
	vops->axpy(N, xa, (tfo - 1.f) / ft + 1.f, xb);
}


#if 0
/**
 * OptISTA
 */
static void optista_theta(int N, float theta[N + 1])
{
	struct ravine_conf conf = ravine_classical;

	theta[0] = 1.;

	for (int i = 1; i < N; i++)
		theta[i] = fista_formula(conf, theta[i - 1]);

	conf.r = 8.;
	theta[N] = fista_formula(conf, theta[N - 1]);
}

static float optista_gamma(int i, int N, const float theta[N + 1])
{
	float thnsq = powf(theta[N], 2.);
	return 2. * theta[i] / thnsq * (thnsq - 2. * powf(theta[i], 2.) + theta[i]);
}
#endif

void landweber_sym(int maxiter, float epsilon, float alpha,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* x, const float* b,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(N);

	double rsnot = vops->norm(N, b);

	for (int i = 0; i < maxiter; i++) {

		iter_monitor(monitor, vops, x);

		iter_op_call(op, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		double rsnew = vops->norm(N, r);

		debug_printf(DP_DEBUG3, "#%d: %f\n", i, rsnew / rsnot);

		if (rsnew < epsilon)
			break;

		vops->axpy(N, x, alpha, r);
	}

	vops->del(r);
}












/**
 * Iterative Soft Thresholding
 *
 * @param maxiter maximum number of iterations
 * @param epsilon stop criterion
 * @param tau (step size) weighting on the residual term, A^H (b - Ax)
 * @param N size of input, x
 * @param vops vector ops definition
 * @param op linear operator, e.g. A
 * @param thresh threshold function, e.g. complex soft threshold
 * @param x initial estimate
 * @param b observations
 * @param monitor compute objective value, errors, etc.
 */
void ist(int maxiter, float epsilon, float tau, long N,
		const struct vec_iter_s* vops,
		ist_continuation_t ist_continuation,
		struct iter_op_s op,
		struct iter_op_p_s thresh,
		float* x, const float* b,
		struct iter_monitor_s* monitor)
{
	struct ist_data itrdata = {

		.rsnew = 1.,
		.rsnot = 1.,
		.iter = 0,
		.maxiter = maxiter,
		.tau = tau,
		.scale = 1.,
	};

	float* r = vops->allocate(N);

	itrdata.rsnot = vops->norm(N, b);


	for (itrdata.iter = 0; itrdata.iter < maxiter; itrdata.iter++) {

		iter_monitor(monitor, vops, x);

		if (NULL != ist_continuation)
			ist_continuation(&itrdata);

		iter_op_call(op, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		itrdata.rsnew = vops->norm(N, r);

		iter_op_p_call(thresh, itrdata.scale * itrdata.tau, x, x);

		debug_printf(DP_DEBUG3, "#It %03d: %f \n", itrdata.iter, itrdata.rsnew / itrdata.rsnot);

		if (itrdata.rsnew < epsilon)
			break;

		vops->axpy(N, x, itrdata.tau, r);
	}

	debug_printf(DP_DEBUG3, "\n");

	vops->del(r);
}


/**
 * Iterative Soft Thresholding/FISTA to solve min || b - Ax ||_2 + lambda * alpha || T x ||_1
 *
 * @param maxiter maximum number of iterations
 * @param epsilon stop criterion
 * @param tau step size
 * @param alpha scaling of prox
 * @param last final application of threshold function
 * @param cf parameters for ravine step
 * @param N size of input, x
 * @param vops vector ops definition
 * @param op linear operator, e.g. A
 * @param thresh threshold function, e.g. complex soft threshold
 * @param x initial estimate
 * @param b observations
 */
void fista(int maxiter, float epsilon, float tau, float alpha,
	bool last,
	struct ravine_conf cf,
	long N,
	const struct vec_iter_s* vops,
	ist_continuation_t ist_continuation,
	struct iter_op_s op,
	struct iter_op_p_s thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor)
{
	struct ist_data itrdata = {

		.rsnew = 1.,
		.rsnot = 1.,
		.iter = 0,
		.maxiter = maxiter,
		.tau = tau,
		.scale = 1.,
	};

	float* r = vops->allocate(N);
	float* o = vops->allocate(N);

	float ra = 1.;
	vops->copy(N, o, x);

	itrdata.rsnot = vops->norm(N, b);


	for (itrdata.iter = 0; itrdata.iter < maxiter; itrdata.iter++) {

		iter_monitor(monitor, vops, x);

		if (NULL != ist_continuation)
			ist_continuation(&itrdata);

		iter_op_p_call(thresh, itrdata.scale * itrdata.tau * alpha, x, x);

		ravine(vops, cf, N, &ra, x, o);	// FISTA
		iter_op_call(op, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		itrdata.rsnew = vops->norm(N, r);

		debug_printf(DP_DEBUG3, "#It %03d: %f   \n", itrdata.iter, itrdata.rsnew / itrdata.rsnot);

		if (itrdata.rsnew < epsilon)
			break;

		vops->axpy(N, x, itrdata.tau, r);
	}

	if (!last) {

		iter_monitor(monitor, vops, x);
		iter_op_p_call(thresh, itrdata.scale * itrdata.tau * alpha, x, x);
	}

	debug_printf(DP_DEBUG3, "\n");
	debug_printf(DP_DEBUG2, "\t\tFISTA iterations: %u\n", itrdata.iter);

	vops->del(o);
	vops->del(r);
}



/**
 *  Landweber L. An iteration formula for Fredholm integral equations of the
 *  first kind. Amer. J. Math. 1951; 73, 615-624.
 */
void landweber(int maxiter, float epsilon, float alpha, long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	float* x, const float* b,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(M);
	float* p = vops->allocate(N);

	double rsnot = vops->norm(M, b);

	for (int i = 0; i < maxiter; i++) {

		iter_monitor(monitor, vops, x);

		iter_op_call(op, r, x);		// r = A x
		vops->xpay(M, -1., r, b);	// r = b - r = b - A x

		double rsnew = vops->norm(M, r);

		debug_printf(DP_DEBUG3, "#%d: %f\n", i, rsnew / rsnot);

		if (rsnew < epsilon)
			break;

		iter_op_call(adj, p, r);
		vops->axpy(N, x, alpha, p);

		if (NULL != callback.fun)
			iter_op_call(callback, x, x);
	}

	vops->del(r);
	vops->del(p);
}


/**
 *
 */
void eulermaruyama(int maxiter, float alpha,
	float step, long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_p_s* thresh,
	float* x, const float* b,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(N);
	float* o = vops->allocate(N);

	for (int i = 0; i < maxiter; i++) {

		iter_monitor(monitor, vops, x);

		// the gradients are scaled so that with unitary operators the
		// data is assumed to have complex Gaussian noise with s = 1
		// (which cancels the 1/2 in the algorithm)

		iter_op_call(op, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		if (thresh) {	// plug&play

			iter_op_p_call(*thresh, step * alpha, o, x);

			vops->axpy(N, x, -1., x);
			vops->axpy(N, x, +1., o);
		}

		vops->axpy(N, x, step, r);

		vops->rand(N, r);
		vops->axpy(N, x, sqrtf(step), r);
	}

	vops->del(o);
	vops->del(r);
}



/**
 * Conjugate Gradient Descent to solve Ax = b for symmetric A
 *
 * @param maxiter maximum number of iterations
 * @param regularization parameter
 * @param epsilon stop criterion
 * @param N size of input, x
 * @param vops vector ops definition
 * @param linop linear operator, i.e. A
 * @param x initial estimate
 * @param b observations
 */
float conjgrad(int maxiter, float l2lambda, float epsilon,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s linop,
	float* x, const float* b,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(N);
	float* p = vops->allocate(N);
	float* Ap = vops->allocate(N);


	// The first calculation of the residual might not
	// be necessary in some cases...

	iter_op_call(linop, r, x);		// r = A x
	vops->axpy(N, r, l2lambda, x);

	vops->xpay(N, -1., r, b);	// r = b - r = b - A x
	vops->copy(N, p, r);		// p = r

	float rsnot = (float)pow(vops->norm(N, r), 2.);
	float rsold = rsnot;
	float rsnew = rsnot;

	float eps_squared = pow(epsilon, 2.);


	int i = 0;

	if (0. == rsold) {

		debug_printf(DP_DEBUG3, "CG: early out\n");
		goto cleanup;
	}

	for (i = 0; i < maxiter; i++) {

		iter_monitor(monitor, vops, x);

		debug_printf(DP_DEBUG3, "#%d: %f\n", i, (double)sqrtf(rsnew));

		iter_op_call(linop, Ap, p);	// Ap = A p
		vops->axpy(N, Ap, l2lambda, p);

		float pAp = (float)vops->dot(N, p, Ap);

		if (0. == pAp)
			break;

		float alpha = rsold / pAp;

		vops->axpy(N, x, +alpha, p);
		vops->axpy(N, r, -alpha, Ap);

		rsnew = pow(vops->norm(N, r), 2.);

		float beta = rsnew / rsold;

		rsold = rsnew;

		if (rsnew <= eps_squared)
			break;

		vops->xpay(N, beta, p, r);	// p = beta * p + r
	}

cleanup:
	vops->del(Ap);
	vops->del(p);
	vops->del(r);

	debug_printf(DP_DEBUG2, "\t cg: %3d\n", i);

	return sqrtf(rsnew);
}


/**
 * Conjugate Gradient Descent to solve Ax = b for blockdiagonal symmetric A
 *
 * @param maxiter maximum number of iterations
 * @param regularization parameter
 * @param epsilon stop criterion
 * @param N size of problem, dims of x: { 2, Bi, N, Bo }
 * @param Bi inner batch size of problem, dims of x: { 2, Bi, N, Bo }
 * @param Bo outer batch size of problem, dims of x: { 2, Bi, N, Bo }
 * @param vops vector ops definition
 * @param linop linear operator, i.e. A
 * @param x initial estimate
 * @param b observations
 */
void conjgrad_batch(int maxiter, float l2lambda, float epsilon,
	long N, long Bi, long Bo,
	const struct vec_iter_s* vops,
	struct iter_op_s linop,
	float* x, const float* b,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(2 * Bo * Bi * N);
	float* p = vops->allocate(2 * Bo * Bi * N);
	float* Ap = vops->allocate(2 * Bo * Bi * N);


	// The first calculation of the residual might not
	// be necessary in some cases...

	iter_op_call(linop, r, x);		// r = A x
	vops->axpy(2 * Bo * Bi * N, r, l2lambda, x);

	vops->xpay(2 * Bo * Bi * N, -1., r, b);	// r = b - r = b - A x
	vops->copy(2 * Bo * Bi * N, p, r);		// p = r

	float* rsnot = vops->allocate(Bo * Bi);
	float* rsold = vops->allocate(Bo * Bi);
	float* rsnew = vops->allocate(Bo * Bi);

	float* pAp = vops->allocate(Bo * Bi);
	float* alpha = vops->allocate(Bo * Bi);
	float* beta = vops->allocate(Bo * Bi);

	vops->dot_bat(Bi, N, Bo, rsnot, r, r);
	vops->copy(Bo * Bi, rsold, rsnot);
	vops->copy(Bo * Bi, rsnew, rsnot);

	vops->smul(Bo * Bi, pow(epsilon, 2.), rsnot, rsnot);

	int i = 0;

	if (0. == vops->norm(Bo * Bi, rsold)) {

		debug_printf(DP_DEBUG3, "CG: early out\n");
		goto cleanup;
	}

	for (i = 0; i < maxiter; i++) {

		iter_monitor(monitor, vops, x);

		debug_printf(DP_DEBUG3, "#%d: %f\n", i, (double)sqrtf(vops->norm(Bo * Bi, rsnew) / (float)(Bo * Bi)));

		iter_op_call(linop, Ap, p);	// Ap = A p
		vops->axpy(Bo * Bi * N, Ap, l2lambda, p);

		vops->dot_bat(Bi, N, Bo, pAp, p, Ap);

		if (0. == vops->norm(Bo * Bi, pAp))
			break;

		vops->div(Bo * Bi, alpha, rsold, pAp);

		vops->axpy_bat(Bi, N, Bo, x, alpha, p);
		vops->smul(Bo * Bi, -1, alpha, alpha);
		vops->axpy_bat(Bi, N, Bo, r, alpha, Ap);

		vops->dot_bat(Bi, N, Bo, rsnew, r, r);

		vops->div(Bo * Bi, beta, rsnew, rsold);

		vops->le(Bo * Bi, rsold, rsnot, rsnew);
		vops->mul(Bo * Bi, rsold, rsold, rsnew);

		if (0. == vops->norm(Bo * Bi, rsold))
			break;

		vops->xpay_bat(Bi, N, Bo, beta, p, r);	// p = beta * p + r
	}

cleanup:
	vops->del(Ap);
	vops->del(p);
	vops->del(r);

	vops->del(rsnot);
	vops->del(rsold);
	vops->del(rsnew);
	vops->del(pAp);
	vops->del(alpha);
	vops->del(beta);

	debug_printf(DP_DEBUG2, "\t cg: %3d\n", i);
}



/**
 * Iteratively Regularized Gauss-Newton Method
 * (Bakushinsky 1993)
 *
 * y = F(x) = F xn + DF dx + ...
 *
 * IRGNM: DF^H ((y - F xn) + DF (xn - x0)) = ( DF^H DF + alpha ) (dx + xn - x0)
 *        DF^H ((y - F xn)) - alpha (xn - x0) = ( DF^H DF + alpha) dx
 *
 * This version only solves the second equation for the update 'dx'. This corresponds
 * to a least-squares problem where the quadratic regularization applies to the difference
 * to 'x0'.
 */
void irgnm(int iter, float alpha, float alpha_min, float redu, long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s adj,
	struct iter_op_p_s inv,
	float* x, const float* xref, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(M);
	float* p = vops->allocate(N);
	float* h = vops->allocate(N);

	for (int i = 0; i < iter; i++) {

		iter_monitor(monitor, vops, x);

		iter_op_call(op, r, x);			// r = F x

		vops->xpay(M, -1., r, y);	// r = y - F x

		debug_printf(DP_DEBUG2, "Step: %u, Res: %f\n", i, vops->norm(M, r));

		iter_op_call(adj, p, r);

		if (NULL != xref)
			vops->axpy(N, p, +alpha, xref);

		vops->axpy(N, p, -alpha, x);

		iter_op_p_call(inv, alpha, h, p);

		vops->axpy(N, x, 1., h);

		alpha = (alpha - alpha_min) / redu + alpha_min;

		if (NULL != callback.fun)
			iter_op_call(callback, x, x);
	}

	vops->del(h);
	vops->del(p);
	vops->del(r);
}



/**
 * Iteratively Regularized Gauss-Newton Method
 * (Bakushinsky 1993)
 *
 * y = F(x) = F xn + DF dx + ...
 *
 * IRGNM: R(DF^H, DF^H DF, alpha) ((y - F xn) + DF (xn - x0)) = (dx + xn - x0)
 *
 * This version has an extra call to DF, but we can use a generic regularized
 * least-squares solver.
 */
void irgnm2(int iter, float alpha, float alpha_min, float alpha_min0, float redu, long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s der,
	struct iter_op_s adj,
	struct iter_op_p_s lsqr,
	float* x, const float* xref, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor)
{

	for (int i = 0; i < iter; i++) {

		iter_monitor(monitor, vops, x);

		float* r = vops->allocate(M);
		iter_op_call(op, r, x);			// r = F x

		vops->xpay(M, -1., r, y);	// r = y - F x

		double* res = vops->norm2(M, r);

		if (NULL != xref)
			vops->axpy(N, x, -1., xref);

		float* q = vops->allocate(M);

		iter_op_call(der, q, x);

		vops->axpy(M, r, +1., q);

		vops->del(q);

		float* a = vops->allocate(N);
		iter_op_call(adj, a, r);
		vops->del(r);

		debug_printf(DP_DEBUG2, "Step: %u, Res: %f\n", i, vops->get_norm2(res));

		iter_op_p_call(lsqr, alpha, x, a);
		vops->del(a);

		if (NULL != xref)
			vops->axpy(N, x, +1., xref);

		alpha = (alpha - alpha_min) / redu + alpha_min;

		if (alpha < alpha_min0)
			alpha = alpha_min0;

		if (NULL != callback.fun)
			iter_op_call(callback, x, x);
	}
}



/**
 * Alternating Minimization
 *
 * Minimize residual by calling each min_op in turn.
 */
void altmin(int iter, float alpha, float redu,
	    long N,
	    const struct vec_iter_s* vops,
	    int NI,
	    struct iter_nlop_s op,
	    struct iter_op_p_s min_ops[NI],
	    float* x[NI], const float* y,
	    struct iter_nlop_s callback)
{
	float* r = vops->allocate(N);
	vops->clear(N, r);


	float* args[1 + NI];
	args[0] = r;

	for (long i = 0; i < NI; ++i)
		args[1 + i] = x[i];

	for (int i = 0; i < iter; i++) {

		for (int j = 0; j < NI; ++j) {

			iter_nlop_call(op, 1 + NI, args); 	// r = F x

			vops->xpay(N, -1., r, y);		// r = y - F x

			debug_printf(DP_DEBUG2, "Step: %u, Res: %f\n", i, vops->norm(N, r));

			iter_op_p_call(min_ops[j], alpha, x[j], y);

			if (NULL != callback.fun)
				iter_nlop_call(callback, NI, x);
		}

		alpha /= redu;
	}

	vops->del(r);
}


/**
 * Projection onto Convex Sets
 *
 * minimize 0 subject to: x in C_1, x in C_2, ..., x in C_D,
 * where the C_i are convex sets
 */
void pocs(int maxiter,
	int D, struct iter_op_p_s proj_ops[static D],
	const struct vec_iter_s* vops,
	long /*N*/, float* x,
	struct iter_monitor_s* monitor)
{
	for (int i = 0; i < maxiter; i++) {

		debug_printf(DP_DEBUG3, "#Iter %d\n", i);

		iter_monitor(monitor, vops, x);

		for (int j = 0; j < D; j++)
			iter_op_p_call(proj_ops[j], 1., x, x); // use temporary memory here?
	}
}


/**
 *  Power iteration
 */
double power(int maxiter,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* u, float* b)
{
	double s = vops->norm(N, u);
	vops->smul(N, 1. / s, u, u);

	if (NULL == b)
		b = u;

	for (int i = 0; i < maxiter; i++) {

		iter_op_call(op, b, u);		// r = A x

		s = vops->norm(N, b);
		vops->smul(N, 1. / s, u, b);
	}

	return s;
}




/**
 * Chambolle Pock First Order Primal Dual algorithm. Solves min_x F(Ax) + G(x)
 *
 * @param maxiter maximum number of iterations
 * @param epsilon stop criterion
 * @param tau primal step size
 * @param sigma dual step size
 * @param sigma_tau_ratio ratio of sigma to tau
 * @param decay decay rate
 * @param adapt_stepsize adaptive step size algorithm
 * @param theta convex combination rate
 * @param N size of input, x
 * @param M array with sizes of transformed input, Ax
 * @param vops vector ops definition
 * @param op_forw array of forward operators, A
 * @param op_adj array of adjoint operators, AH
 * @param prox1 array of proximal functions of F, e.g. prox_l2bal
 * @param prox2 proximal function of G, e.g. prox_wavelet_thresh
 * @param x initial estimate
 * @param monitor callback function
 */
void chambolle_pock(float alpha, int maxiter, float epsilon, float tau, float sigma, 
	float sigma_tau_ratio, float theta, float decay, bool adapt_stepsize,
	int O, long N, long M[O],
	const struct vec_iter_s* vops,
	struct iter_op_s op_norm,
	struct iter_op_s op_forw[O],
	struct iter_op_s op_adj[O],
	struct iter_op_p_s prox1[O],
	struct iter_op_p_s prox2,
	float* x, const float* xadj,
	struct iter_monitor_s* monitor)
{
	float* x_avg = vops->allocate(N);
	vops->copy(N, x_avg, x);

	float* u[O];
	for (int j = 0; j < O; j++) {

		u[j] = vops->allocate(M[j]);
		vops->clear(M[j], u[j]);
	}

	float* Ahu = NULL;

	if (NULL != xadj) {

		Ahu = vops->allocate(N);
		vops->clear(N, Ahu);
	}

	for (int i = 0; i < maxiter; i++) {

		float lambda = (float)pow(decay, i);

		/* update u
		 * u0 = u
		 * p = u + sigma * A(x)
		 * u = p - sigma * prox1(p / sigma, 1 / sigma)
		 * u = lambda * u + (1 - lambda) * u0
		 */

		float res2 = 0;

		if (NULL != xadj) {

			float* Ahu_old = vops->allocate(N);
			float* Ahu_new = vops->allocate(N);

			iter_op_call(op_norm, Ahu_old, x_avg);

			vops->xpay(N, sigma, Ahu_old, Ahu);

			vops->axpbz(N, Ahu_new, 1. / (1. + sigma), Ahu_old, -sigma / (1. + sigma), xadj);

			vops->copy(N, Ahu_old, Ahu);
			vops->axpbz(N, Ahu, lambda, Ahu_new, 1. - lambda, Ahu_old);

			vops->sub(N, Ahu_old, Ahu, Ahu_old);

			// This is different to the norm in the loop below
			double* tmp_snorm = vops->dot2(N, Ahu_old, Ahu_old);

			vops->del(Ahu_new);
			vops->del(Ahu_old);

			res2 += vops->get_dot2(tmp_snorm);
		}


		for (int j = 0; j < O; j++) {

			float* u_old = vops->allocate(M[j]);
			float* u_new = vops->allocate(M[j]);

			iter_op_call(op_forw[j], u_old, x_avg);

			vops->axpy(M[j], u_old, 1. / sigma, u[j]); // (u + sigma * A(x)) / sigma

			iter_op_p_call(prox1[j], 1. / sigma * alpha, u_new, u_old);

			vops->axpbz(M[j], u_new, -1. * sigma, u_new, sigma, u_old);
			vops->copy(M[j], u_old, u[j]);
			vops->axpbz(M[j], u[j], lambda, u_new, 1. - lambda, u_old);

			vops->sub(M[j], u_old, u[j], u_old);

			double* tmp_snorm = vops->dot2(M[j], u_old, u_old);

			vops->del(u_old);
			vops->del(u_new);

			res2 += vops->get_dot2(tmp_snorm);
		}

		res2 = sqrtf(res2) / tau;

		/* update x
		 * x0 = x
		 * q = x0 - tau * AH(u)
		 * x = prox2(q, tau)
		 * x = lambda * x + (1 - lambda * x0)
		 */

		float* x_old = vops->allocate(N);
		float* x_new = vops->allocate(N);

		vops->copy(N, x_old, x);

		if (NULL != Ahu)
			vops->axpy(N, x, -1. * tau, Ahu);

		for (int j = 0; j < O; j++) {

			iter_op_call(op_adj[j], x_new, u[j]);
			vops->axpy(N, x, -1. * tau, x_new);
		}

		iter_op_p_call(prox2, tau * alpha, x_new, x);

		vops->axpbz(N, x, lambda, x_new, 1. - lambda, x_old);

		/* adapt step sizes
		* x_new = x - x_old
		* u = A( x_new )
		* norm_Kx = || u ||_2
		* norm_x = || x_new ||_2
		*/
		if (adapt_stepsize) {
			
			vops->sub(N, x_new, x, x_old);

			float norm_Kx = 0;
			
			// || A( x - x_old ) ||_2

			for (int j = 0; j < O; j++) {

				float* u = vops->allocate(M[j]);

				iter_op_call(op_forw[j], u, x_new);
				norm_Kx += pow(vops->norm(M[j], u), 2.);

				vops->del(u);
			}

			if (NULL != Ahu) {

				float* Ahu_new = vops->allocate(N);

				iter_op_call(op_norm, Ahu_new, x_new);
				norm_Kx += vops->dot(N, Ahu_new, x_new);

				vops->del(Ahu_new);
			}

			norm_Kx = sqrt(norm_Kx);

			// || ( x - x_old ) ||_2

			float norm_x = vops->norm(N, x_new);

			if (0 != norm_Kx) {

				float ratio = norm_x / norm_Kx;
				
				float sigma_tau_sqrt = sqrtf(sigma * tau);

				float threshold = 0.95f * sigma_tau_sqrt;

				float temp;

				// adapt step sizes depending on ratio

				if ((ratio < sigma_tau_sqrt) && (0 != ratio)) {

					if (threshold < ratio)
						temp = threshold;
					else
						temp = ratio;
				} else {

					temp = sigma_tau_sqrt;
				}
				
				sigma = temp * sigma_tau_ratio;
				tau = temp / sigma_tau_ratio;

				debug_printf(DP_DEBUG3, "#Step sizes %03d: sigma: %f, tau: %f  \n", i, sigma, tau);

			} else {
				
				debug_printf(DP_DEBUG3, "#Step sizes unchanged.\n");
			}
		}

		/* update x_avg
		 * a_avg = x + theta * (x - x0)
		 */
		vops->axpbz(N, x_avg, 1 + theta, x, -1. * theta, x_old);

		// residual
		vops->sub(N, x_old, x, x_old);
		float res1 = vops->norm(N, x_old) / sigma;

		vops->del(x_old);
		vops->del(x_new);

		iter_monitor(monitor, vops, x);

		debug_printf(DP_DEBUG3, "#It %03d: %f %f  \n", i, res1, res2);

		if (epsilon > (res1 + res2))
			break;
	}

	debug_printf(DP_DEBUG3, "\n");

	vops->del(x_avg);

	for (int j = 0; j < O; j++)
		vops->del(u[j]);

	if (NULL != Ahu)
		vops->del(Ahu);
}



/**
 * Compute the sum of the selected outputs, selected outputs must be scalars
 *
 * @param NO number of outputs of nlop
 * @param NI number of inputs of nlop
 * @param nlop nlop to apply
 * @param args out- and inputs of operator
 * @param out_optimize_flag sums outputs over selected outputs, selected outputs must be scalars
 * @param der_in_flag only information to compute derivatives with respect to selected inputs are stores
 * @param vops vector operators
 **/
static float compute_objective(int NO, int NI, struct iter_nlop_s nlop, float* args[NO + NI], unsigned long out_optimize_flag, unsigned long der_in_flag, const struct vec_iter_s* vops)
{
	float result = 0;
	iter_nlop_call_select_der(nlop, NO + NI, args, out_optimize_flag, der_in_flag); 	// r = F x

	for (int o = 0; o < NO; o++) {
		if (MD_IS_SET(out_optimize_flag, o)) {

			float tmp;
			vops->copy(1, &tmp, args[o]);
			result += tmp;
		}
	}

	return result;
}

/**
 * Compute the gradient with respect to the inputs selected by in_optimize_flag.
 * The result is the sum of the gradients with respect to the outputs selected by out_optimize_flag
 *
 * @param NI number of inputs of nlop
 * @param in_optimize_flag compute gradients with respect to selected inputs
 * @param isize sizes of input tensors
 * @param grad output of the function, grad[i] must be allocated for selected inputs
 * @param NO number of outputs of nlop
 * @param out_optimize_flag sums gradients over selected outputs, selected outputs must be scalars
 * @param adj array of adjoint operators
 * @param vops vector operators
 **/
static void getgrad(int NI, unsigned long in_optimize_flag, long isize[NI], float* grad[NI], int NO, unsigned long out_optimize_flag, struct iter_op_arr_s adj, const struct vec_iter_s* vops)
{
	float* one = vops->allocate(2);
	float one_var[2] = { 1., 0. };	// complex
	const float* one_arr[] = { one };

	vops->copy(2, one, one_var);

	float* tmp_grad[NI];

	for (int i = 0; i < NI; i++) // analyzer false positive
		tmp_grad[i] = NULL;

	for (int i = 0; i < NI; i++)
		if ((1 < NO) && MD_IS_SET(in_optimize_flag, i))
			tmp_grad[i] = vops->allocate(isize[i]);

	for (int o = 0, count = 0; o < NO; o++) {

		if (!MD_IS_SET(out_optimize_flag, o))
			continue;

		iter_op_arr_call(adj, NI, in_optimize_flag, (0 == count) ? grad : tmp_grad, 1, MD_BIT(o), one_arr);

		for (int i = 0; i < NI; i++)
			if ((0 < count) && MD_IS_SET(in_optimize_flag, i))
				vops->add(isize[i], grad[i], grad[i], tmp_grad[i]);
		count += 1;
	}

	for (int i = 0; i < NI; i++)
		if ((1 < NO) && MD_IS_SET(in_optimize_flag, i))
			vops->del(tmp_grad[i]);

	vops->del(one);
}


/**
 * Prototype for sgd-like algorithm
 * The gradient is computed and the operator "update" computes the update, this operator can remember information such as momentum
 *
 * @param epochs number of epochs to train (one epoch corresponds to seeing each dataset once)
 * @param batches number of updates per epoch
 * @param learning_rate (overwritten by learning_rate_schedule if != NULL)
 * @param batchnorm_momentum momentum for updating mean and variance of batch normalization
 * @param learning_rate_schedule learning rate for each update
 * @param NI number of input tensors
 * @param isize size of input tensors (flattened as real)
 * @param in_type type of inputs (static, batchgen, to optimize)
 * @param x inputs of operator (weights, train data, reference data)
 * @param NO number of output tensors (i.e. objectives)
 * @param osize size of output tensors (flattened as real)
 * @param out_type type of output (i.e. should be minimized)
 * @param N_batch batch size
 * @param N_total total size of datasets
 * @param vops
 * @param nlop nlop for minimization
 * @param adj array of adjoints of nlop
 * @param prox prox operators applied after each update on the current weights
 * @param nlop_batch_gen nlop for generating a new batch for each update
 * @param update diagonal array of operator computing the update based on the gradient
 * @param callback UNUSED
 * @param monitor UNUSED
 */
void sgd(	int epochs, int batches,
		float learning_rate, float batchnorm_momentum,
		const float (*learning_rate_schedule)[epochs][batches],
		int NI, long isize[NI], enum IN_TYPE in_type[NI], float* x[NI],
		int NO, long osize[NO], enum OUT_TYPE out_type[NI],
		int N_batch, int N_total,
		const struct vec_iter_s* vops,
		struct iter_nlop_s nlop, struct iter_op_arr_s adj,
		struct iter_op_p_s update[NI],
		struct iter_op_p_s prox[NI],
		struct iter_nlop_s nlop_batch_gen,
		struct iter_op_s /*callback*/,
		struct monitor_iter6_s* monitor, const struct iter_dump_s* dump)
{
	float* grad[NI];
	float* dxs[NI];
	float* args[NO + NI];

	float* x_batch_gen[NI]; //arrays which are filled by batch generator
	long N_batch_gen = 0;

	unsigned long in_optimize_flag = 0;
	unsigned long out_optimize_flag = 0;

	if (batches != N_total / N_batch)
		error("Wrong number of batches!\n");

	for (int i = 0; i < NI; i++) {

		grad[i] = NULL;
		dxs[i] = NULL;

		switch (in_type[i]) {

		case IN_STATIC:
		case IN_BATCH:
		case IN_BATCHNORM:

			break;

		case IN_OPTIMIZE:

			grad[i] = vops->allocate(isize[i]);
			dxs[i] = vops->allocate(isize[i]);

			in_optimize_flag = MD_SET(in_optimize_flag, i);

			if (NULL != prox[i].fun)
				iter_op_p_call(prox[i], 0, x[i], x[i]); //project to constraint

			break;

		case IN_BATCH_GENERATOR:

			if (NULL != x[i])
				error("NULL != x[%d] for batch generator\n", i);

			x[i] = vops->allocate(isize[i]);

			x_batch_gen[N_batch_gen] = x[i];
			N_batch_gen += 1;
			break;

		default:

			error("unknown flag\n");
			break;
		}

		args[NO + i] = x[i];
	}

	for (int o = 0; o < NO; o++) {

		args[o] = vops->allocate(osize[o]);

		if (OUT_OPTIMIZE == out_type[o])
			out_optimize_flag = MD_SET(out_optimize_flag, o);
	}

	const float *x2[NI];
	for (int i = 0; i < NI; i++)
		x2[i] = x[i];

	for (int epoch = 0; epoch < epochs; epoch++) {

		iter_dump(dump, epoch, NI, x2);

		for (int i_batch = 0; i_batch < N_total / N_batch; i_batch++) {

			if (0 != N_batch_gen)
				iter_nlop_call(nlop_batch_gen, N_batch_gen, x_batch_gen);

			float r0 = compute_objective(NO, NI, nlop, args, out_optimize_flag, in_optimize_flag, vops); // update graph and compute loss

			getgrad(NI, in_optimize_flag, isize, grad, NO, out_optimize_flag, adj, vops);

			int batchnorm_counter = 0;

			if (NULL != learning_rate_schedule)
				learning_rate = (*learning_rate_schedule)[epoch][i_batch];

			for (int i = 0; i < NI; i++) {

				if (in_type[i] == IN_OPTIMIZE) {

					iter_op_p_call(update[i], learning_rate, dxs[i], grad[i]);

					vops->add(isize[i], args[NO + i], args[NO + i], dxs[i]);

					if (NULL != prox[i].fun)
						iter_op_p_call(prox[i], learning_rate, args[NO + i], args[NO + i]);
				}

				if (in_type[i] == IN_BATCH)
					args[NO + i] += isize[i];

				if (in_type[i] == IN_BATCHNORM) {

					int o = 0;
					int j = batchnorm_counter;

					while ((OUT_BATCHNORM != out_type[o]) || (j > 0)) {

						if (OUT_BATCHNORM == out_type[o])
							j--;
						o++;
					}

					vops->smul(isize[i], batchnorm_momentum, x[i], x[i]);
					vops->axpy(isize[i], x[i],  1. - batchnorm_momentum, args[o]);

					batchnorm_counter++;
				}
			}

			monitor_iter6(monitor, epoch, i_batch, N_total / N_batch, r0, NI, x2, NULL);
		}

		for (int i = 0; i < NI; i++)
			if (in_type[i] == IN_BATCH)
				args[NO + i] -= isize[i] * (N_total / N_batch);
	}

	for (int i = 0; i < NI; i++) {

		if (NULL != grad[i])
			vops->del(grad[i]);

		if (NULL != dxs[i])
			vops->del(dxs[i]);

		if (IN_BATCH_GENERATOR == in_type[i]) {

			vops->del(x[i]);
			x[i] = NULL;
		}
	}

	for (int o = 0; o < NO; o++)
		if (NULL != args[o])
			vops->del(args[o]);
}

/**
 * iPALM: Inertial Proximal Alternating Linearized Minimization.
 * Solves min_{x_0, ..., x_N} H({x_0, ..., x_N}) + sum_i f_i(x_i)
 * https://doi.org/10.1137/16M1064064
 *
 * kth iteration step for input i:
 *
 * y_i^k := x_i^k + alpha_i^k (x_i^k - x_i^{k-1})
 * z_i^k := x_i^k + beta_i^k (x_i^k - x_i^{k-1})
 * x_i^{k+1} = prox^{f_i}_{tau_i} (y_i^k - 1/tau_i grad_{x_i} H(x_0^{k+1}, ... z_i^k, x_{i+1}^k, ...))
 *
 * @param NI number of input tensors
 * @param isize size of input tensors (flattened as real)
 * @param in_type type of inputs (static, batchgen, to optimize)
 * @param x inputs of operator (weights, train data, reference data)>
 * @param x_old weights of the last iteration (is initialized if epoch_start == 0)
 * @param NO number of output tensors (i.e. objectives)
 * @param osize size of output tensors (flattened as real)
 * @param out_type type of output (i.e. should be minimized)
 * @param N_batch number of batches per epoch
 * @param epoch_start warm start possible if epoch start > 0, note that epoch corresponds to an update due to one batch
 * @param epoch_end
 * @param vops
 * @param alpha parameter per input
 * @param beta parameter per input
 * @param convex parameter per input, determines stepsize
 * @param L Lipschitz constants
 * @param Lmin minimal Lipschitz constant for backtracking
 * @param Lmax maximal Lipschitz constant for backtracking
 * @param Lshrink L->L / L_shrinc if Lipschitz condition is satisfied
 * @param Lincrease L->L * Lincrease if Lipschitz condition is not satisfied
 * @param nlop nlop for minimization
 * @param adj array of adjoints of nlop
 * @param prox proximal operators of f, if (NULL == prox[i].fun) f = 0 is assumed
 * @param nlop_batch_gen operator copying current batch in inputs x[i] with type batch generator
 * @param callback UNUSED
 * @param monitor UNUSED
 */
void iPALM(	long NI, long isize[NI], enum IN_TYPE in_type[NI], float* x[const NI], float* x_old[NI],
		long NO, long osize[NO], enum OUT_TYPE out_type[NO],
		int N_batch, int epoch_start, int epoch_end,
		const struct vec_iter_s* vops,
		float alpha[NI], float beta[NI], bool convex[NI], bool trivial_stepsize, bool reduce_momentum,
		float L[NI], float Lmin, float Lmax, float Lshrink, float Lincrease,
		struct iter_nlop_s nlop,
		struct iter_op_arr_s adj,
		struct iter_op_p_s prox[NI],
		float batchnorm_momentum,
		struct iter_nlop_s nlop_batch_gen,
		struct iter_op_s /*callback*/,
		struct monitor_iter6_s* monitor, const struct iter_dump_s* dump)
{
	float* x_batch_gen[NI]; //arrays which are filled by batch generator
	long N_batch_gen = 0;

	float* args[NO + NI];

	float* x_new[NI];
	float* y[NI];
	float* z[NI];
	float* tmp[NI];
	float* grad[NI];

	unsigned long out_optimize_flag = 0;

	for (int i = 0; i < NI; i++) {

		x_batch_gen[i] = NULL;

		x_new[i] = NULL;
		y[i] = NULL;
		z[i] = NULL;
		tmp[i] = NULL;
		grad[i] = NULL;

		switch (in_type[i]){

		case IN_STATIC:
			break;

		case IN_BATCH:

			error("flag IN_BATCH not supported\n");
			break;

		case IN_OPTIMIZE:

			if (0 != epoch_start)
				break;

			if (NULL != prox[i].fun) {

				iter_op_p_call(prox[i], 0., x_old[i], x[i]); // if prox is a projection, we apply it, else it is just a copy (mu = 0)

				vops->copy(isize[i], x[i], x_old[i]);

			} else {

				vops->copy(isize[i], x_old[i], x[i]);
			}

			break;

		case IN_BATCH_GENERATOR:

			if (NULL != x[i])
				error("NULL != x[%d] for batch generator\n", i);

			x[i] = vops->allocate(isize[i]);

			x_batch_gen[N_batch_gen] = x[i];
			N_batch_gen += 1;

			break;

		case IN_BATCHNORM:

			break;

		default:

			error("unknown flag\n");
			break;
		}

		args[NO + i] = x[i];
	}

	for (int o = 0; o < NO; o++) {

		args[o] = vops->allocate(osize[o]);

		if (OUT_OPTIMIZE == out_type[o])
			out_optimize_flag = MD_SET(out_optimize_flag, o);
	}

	const float *x2[NI];
	for (int i = 0; i < NI; i++)
		x2[i] = x[i];

	for (int epoch = epoch_start; epoch < epoch_end; epoch++) {

		iter_dump(dump, epoch, NI, x2);

		for (int batch = 0; batch < N_batch; batch++) {

			if (0 != N_batch_gen)
				iter_nlop_call(nlop_batch_gen, N_batch_gen, x_batch_gen);

			float r_old = compute_objective(NO, NI, nlop, args, out_optimize_flag, 0, vops);

			float r_i = r_old;

			for (int i = 0; i < NI; i++) {

				if (IN_OPTIMIZE != in_type[i])
					continue;

				grad[i] = vops->allocate(isize[i]);
				tmp[i] = vops->allocate(isize[i]);
				y[i] = vops->allocate(isize[i]);
				x_new[i] = vops->allocate(isize[i]);

				//determine current parameters
				float betai = (-1. == beta[i]) ? (float)(epoch * N_batch + batch) / (float)((epoch * N_batch + batch) + 3.) : beta[i];
				float alphai = (-1. == alpha[i]) ? (float)(epoch * N_batch + batch) / (float)((epoch * N_batch + batch) + 3.) : alpha[i];

				float r_z = 0;

				if (!reduce_momentum) {

					//Compute gradient at z = x^n + alpha * (x^n - x^(n-1))
					z[i] = vops->allocate(isize[i]);

					vops->axpbz(isize[i], z[i], 1 + betai, x[i], -betai, x_old[i]); // tmp1 = z = x^n + alpha * (x^n - x^(n-1))

					args[NO + i] = z[i];

					r_z = compute_objective(NO, NI, nlop, args, out_optimize_flag, MD_BIT(i), vops);

					vops->del(z[i]);

					getgrad(NI, MD_BIT(i), isize, grad, NO, out_optimize_flag, adj, vops);
				}

				//backtracking
				bool lipshitz_condition = false;
				float reduce_momentum_scale = 1;

				while (!lipshitz_condition) {

					if (reduce_momentum) {

						//Compute gradient at z = x^n + alpha * (x^n - x^(n-1))
						z[i] = vops->allocate(isize[i]);

						vops->axpbz(isize[i], z[i], 1 + reduce_momentum_scale * betai, x[i], -(reduce_momentum_scale * betai), x_old[i]); // tmp1 = z = x^n + alpha * (x^n - x^(n-1))

						args[NO + i] = z[i];

						r_z = compute_objective(NO, NI, nlop, args, out_optimize_flag, MD_BIT(i), vops);

						vops->del(z[i]);

						getgrad(NI, MD_BIT(i), isize, grad, NO, out_optimize_flag, adj, vops);
					}


					float tau = convex[i] ? (1. + 2. * betai) / (2. - 2. * alphai) * L[i] : (1. + 2. * betai) / (1. - 2. * alphai) * L[i];

					if (trivial_stepsize || (-1. == beta[i]) || (-1. == alpha[i]))
						tau = L[i];

					if ((0 > betai) || ( 0 > alphai) || ( 0 > tau))
						error("invalid parameters alpha[%d]=%f, beta[%d]=%f, tau=%f\n", i, alphai, i, betai, tau);

					//compute new weights
					vops->axpbz(isize[i], y[i], 1 + reduce_momentum_scale * alphai, x[i], -(reduce_momentum_scale * alphai), x_old[i]);
					vops->axpbz(isize[i], tmp[i], 1, y[i], -1./tau, grad[i]); //tmp2 = x^n + alpha*(x^n - x^n-1) - 1/tau grad

					if (NULL != prox[i].fun)
						iter_op_p_call(prox[i], tau, x_new[i], tmp[i]);
					else
						vops->copy(isize[i],  x_new[i], tmp[i]);

					//compute new residual
					args[NO + i] = x_new[i];

					float r_new = compute_objective(NO, NI, nlop, args, out_optimize_flag, 0, vops);

					//compute Lipschitz condition at z
					float r_lip_z = r_z;

					vops->sub(isize[i], tmp[i], x_new[i], y[i]); // tmp = x^(n+1) - y^n

					r_lip_z += vops->dot(isize[i], grad[i], tmp[i]);
					r_lip_z += L[i] / 2. * vops->dot(isize[i], tmp[i], tmp[i]);

					if ((r_lip_z * 1.001 >= r_new) || (L[i] >= Lmax)) { //1.001 for flp errors

						lipshitz_condition = true;

						if (L[i] > Lmin)
							L[i] /= Lshrink;

						vops->copy(isize[i], x_old[i], x[i]);
						vops->copy(isize[i], x[i], x_new[i]);

						r_i = r_new; //reuse the new residual within one batch (no update of training data)

					} else {

						reduce_momentum_scale /= Lincrease;
						L[i] *= Lincrease;
					}
				}

				args[NO + i] = x[i];

				vops->del(grad[i]);
				vops->del(tmp[i]);
				vops->del(y[i]);
				vops->del(x_new[i]);

				grad[i] = NULL;
				tmp[i] = NULL;
				y[i] = NULL;
				z[i] = NULL;
				x_new[i] = NULL;

				int batchnorm_counter = 0;

				for (int i = 0; i < NI; i++) {

					if (in_type[i] == IN_BATCHNORM) {

						int o = 0;
						int j = batchnorm_counter;

						while ((OUT_BATCHNORM != out_type[o]) || (j > 0)) {

							if (OUT_BATCHNORM == out_type[o])
								j--;
							o++;
						}

						vops->smul(isize[i], batchnorm_momentum, x[i], x[i]);
						vops->axpy(isize[i], x[i],  1. - batchnorm_momentum, args[o]);

						batchnorm_counter++;
					}
				}
			}

			// FIXME:
			char post_string[20 * (NI?:1)];
			sprintf(post_string, " ");

			for (int i = 0; i < NI; i++)
				if (IN_OPTIMIZE == in_type[i])
					sprintf(post_string + strlen(post_string), "L[%d]=%.3e ", i, L[i]);

			monitor_iter6(monitor, epoch, batch, N_batch, r_i, NI, x2, post_string);
		}
	}


	for (int i = 0; i < NI; i++) {

		if (IN_BATCH_GENERATOR == in_type[i]) {

			vops->del(x[i]);
			x[i] = NULL;
		}
	}

	for (int o = 0; o < NO; o++)
		if (NULL != args[o])
			vops->del(args[o]);
}


static float compute_grad_obj(struct iter_op_s op, struct iter_op_s adj, float* grad, const float* arg, const struct vec_iter_s* vops)
{
	float result = 0;
	float* tmp = vops->allocate(2);

	iter_op_call(op, tmp, arg);
	vops->copy(1, &result, tmp);

	float one_var[2] = { 1., 0. };	// complex
	vops->copy(2, tmp, one_var);

	iter_op_call(adj, grad, tmp);
	vops->del(tmp);

	return result;
}


static bool line_search_backtracking(struct iter_op_s op, struct iter_op_s adj, const struct vec_iter_s* vops,
		 int N, float x[N], const float xprev[N], float g[N], const float p[N],
		 float* f, float* stp,
		 float c1, float c2)
{
	const float dec = 0.5;
	const float inc = 2.1;
	const float stp_max = 1.e10;
	const float stp_min = 1.e-10;
	const int max_iter = 50;

	bool armijo = (0 >= c2); //wolfe condition else

	if (0. >= *stp)
		error("Non-positive step size!\n");

	/* Compute the initial gradient in the search direction. */
	float finit = *f;
	float dginit = vops->dot(N, g, p);
	float dgtest = c1 * dginit;
	float fprev = 0;


	if (0 < dginit)
		error("Non-decreasing search direction!\n");

	for (int i = 0; i < max_iter; i++) {

		vops->axpbz(N, x, 1., xprev, *stp, p);

		*f = compute_grad_obj(op, adj, g, x, vops);

		float width;

		if (*f > finit + *stp * dgtest) {

			if ((i > 0) && (fabsf(finit - *f) < 1.e-6 * MAX(fabsf(finit), fabsf(*f))) && (fabsf(fprev - *f) < 1.e-6 * MAX(fabsf(fprev), fabsf(*f))))
				return false;

			width = dec;
		} else {

			if (armijo)
				return true;

			/* Check the Wolfe condition. */
			float dg = vops->dot(N, g, p);

			if (dg < c2 * dginit)
				width = inc;
			else
				return true;
		}

		if (*stp < stp_min)
			return false;

		if (*stp > stp_max)
			error("Backtracking maximum step size reached!\n");

		(*stp) *= width;

		fprev = *f;
	}

	return false;
}

#if 0
//Algorithm 3.6 in Numerical Optimization by Jorge Nocedal & Stephen J. Wright
static bool zoom(struct iter_op_s op, struct iter_op_s adj, const struct vec_iter_s* vops,
		 int N,
		 float x[N], const float xprev[N],
		 float g[N],  const float p[N],
		 float phi0, float phip0, float phi_lo,
		 float* f, float* stp,
		 float stp_hi, float stp_lo, float c1, float c2)
{
	for (int j = 0; j < 20; j++) {

		if (fabsf(stp_hi - stp_lo) / MAX(stp_hi, stp_lo) < 1.e-5)
			return true;

		if (1.e-20 > MAX(stp_hi, stp_lo))
			return false;

		*stp = (stp_hi + stp_lo) / 2.;

		vops->axpbz(N, x, 1., xprev, *stp, p);
		*f = compute_grad_obj(op, adj, g, x, vops);

		if ((*f > phi0 + c1 * (*stp) * phip0) || (*f >= phi_lo)) {

			stp_hi = *stp;
		} else {

			float phip = vops->dot(N, g, p);

			if (fabsf(phip) <= -c2 * phip0)
				return true;

			if (phip * (stp_hi - stp_lo) >= 0)
				stp_hi = stp_lo;

			*stp = stp_hi;
			phi_lo = *f;
		}
	}

	debug_printf(DP_WARN,"L-BFGS: Maximum recursion in zoom! %e %e\n", stp_lo, stp_hi);
	return false;
}

//Algorithm 3.6 in Numerical Optimization by Jorge Nocedal & Stephen J. Wright
//after return f, x, and g will be updated
static bool stepsize(struct iter_op_s op, struct iter_op_s adj, const struct vec_iter_s* vops,
		 int N, float x[N], const float xprev[N], float g[N], const float p[N],
		 float* f, float* stp,
		 float c1, float c2)
{
	float aim1 = 0;

	float phi0 = *f;
	float phiaim1 = *f;

	float phip0 = vops->dot(N, g, p);

	for (int i = 1; i < 20; i++) {

		vops->axpbz(N, x, 1., xprev, *stp, p);

		float phiai = compute_grad_obj(op, adj, g, x, vops);

		if ((phiai > phi0 + c1 * (*stp) * phip0) || ((phiai >= phiaim1) && (1 < i))) {

			return zoom(op, adj, vops, N, x, xprev, g, p, phi0, phip0, phiaim1, f, stp, *stp, aim1, c1, c2);
		}

		float phipai = vops->dot(N, g, p);

		if (fabsf(phipai) <= -c2 * phip0)
			return true;

		if (0 <= phipai)
			return zoom(op, adj, vops, N, x, xprev, g, p, phi0, phip0, phiai, f, stp, aim1, *stp, c1, c2);

		if (*stp > 1.e10)
			return false;

		aim1 = *stp;
		phiaim1 = phiai;
		(*stp) *=  2;
	}

	debug_printf(DP_WARN,"L-BFGS: Maximum recursion in stepsize estimation!\n");
	return false;
}

#endif

void lbfgs(int maxiter, int M, float step, float ftol, float gtol, float c1, float c2, struct iter_op_s op, struct iter_op_s adj, int N, float *x, const struct vec_iter_s* vops)
{
	float* y[M];
	float* s[M];
	float rho[M];

	float* r = vops->allocate(N);

	for (int i = 0; i < M; i++) {

		y[i] = vops->allocate(N);;
		s[i] = vops->allocate(N);;
		rho[i] = 0;
	}

	float* gprev = vops->allocate(N);
	float* xprev = vops->allocate(N);

	float* g = vops->allocate(N);
	float f = compute_grad_obj(op, adj, g, x, vops);

	float* p = vops->allocate(N);

	for (int k = 0; k < maxiter || -1 == maxiter; k++) {

		debug_printf(DP_DEBUG1, "L-BFGS iter %d: obj: %e!\n", k, f);

		if (gtol > vops->norm(N, g) / MAX(1., vops->norm(N, x))) {

			debug_printf(DP_DEBUG1, "L-BFGS converged after %d iterations\n", k);
			break;
		}

		float q[N];
		vops->copy(N, q, g);

		float alpha[M];

		for (int ip = k - 1; ip >= MAX(k - M, 0); ip--) {

			int i = ip % M;

			alpha[i] = rho[i] * vops->dot(N, s[i], q);
			vops->axpy(N, q, -alpha[i], y[i]);
		}

		float gamma = 1;

		if (0 < k) {

			float num = vops->dot(N, s[(k - 1) % M], y[(k - 1) % M]);
			float den = vops->dot(N, y[(k - 1) % M], y[(k - 1) % M]);

			if ((0. >= num) || (0. == den))
				gamma = 1.;
			else
				gamma = num / den;
		}

		vops->smul(N, gamma, r, q);

		for (int ip = MAX(k - M, 0); ip < k; ip++) {

			int i = ip % M;

			float beta = rho[i] * vops->dot(N, y[i], r);
			vops->axpy(N, r, alpha[i] - beta, s[i]);
		}

		vops->smul(N, -1, p, r);

		vops->copy(N, xprev, x);
		vops->copy(N, gprev, g);

		float fprev = f;

		if (!line_search_backtracking(op, adj, vops, N, x, xprev, g, p, &f, &step, c1, c2)) {

			debug_printf(DP_DEBUG1, "L-BFGS terminated after %d iterations as no new stepsize could be found (stp = %e)!\n", k, step);

			vops->copy(N, x, xprev);
			vops->copy(N, g, gprev);
			break;
		}

		if (ftol >= (fprev - f) / MAX(1., MAX(fabsf(fprev), fabsf(f)))) {

			debug_printf(DP_DEBUG1, "L-BFGS converged after %d iterations with fopt=%e\n", k + 1, f);
			break;
		}

		vops->sub(N, s[k % M], x, xprev);
		vops->sub(N, y[k % M], g, gprev);

		rho[k % M] = 1. / vops->dot(N, y[k % M], s[k % M]);


		if (0. >= c2)
			step = 1.; //no increase in step size for armijo condition
	}

	vops->del(r);
	vops->del(gprev);
	vops->del(xprev);
	vops->del(g);
	vops->del(p);

	for (int i = 0; i < M; i++) {

		vops->del(y[i]);
		vops->del(s[i]);
	}
}

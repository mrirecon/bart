/* Copyright 2013-2017. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
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
 */

#include <math.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "iter/vec.h"
#include "iter/monitor.h"

#include "italgos.h"

extern inline void iter_op_call(struct iter_op_s op, float* dst, const float* src);
extern inline void iter_nlop_call(struct iter_nlop_s op, int N, float* args[N]);
extern inline void iter_op_p_call(struct iter_op_p_s op, float rho, float* dst, const float* src);

/**
 * ravine step
 * (Nesterov 1983)
 */
static void ravine(const struct vec_iter_s* vops, long N, float* ftp, float* xa, float* xb)
{
	float ft = *ftp;
	float tfo = ft;

	ft = (1.f + sqrtf(1.f + 4.f * ft * ft)) / 2.f;
	*ftp = ft;

	vops->swap(N, xa, xb);
	vops->axpy(N, xa, (1.f - tfo) / ft - 1.f, xa);
	vops->axpy(N, xa, (tfo - 1.f) / ft + 1.f, xb);
}









void landweber_sym(unsigned int maxiter, float epsilon, float alpha, long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* x, const float* b,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(N);

	double rsnot = vops->norm(N, b);

	for (unsigned int i = 0; i < maxiter; i++) {

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
void ist(unsigned int maxiter, float epsilon, float tau, long N,
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
		
		iter_op_p_call(thresh, itrdata.scale * itrdata.tau, x, x);


		iter_op_call(op, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		itrdata.rsnew = vops->norm(N, r);

		debug_printf(DP_DEBUG3, "#It %03d: %f \n", itrdata.iter, itrdata.rsnew / itrdata.rsnot);

		if (itrdata.rsnew < epsilon)
			break;

		vops->axpy(N, x, itrdata.tau, r);
	}

	debug_printf(DP_DEBUG3, "\n");

	vops->del(r);
}



/**
 * Iterative Soft Thresholding/FISTA to solve min || b - Ax ||_2 + lambda || T x ||_1
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
 */
void fista(unsigned int maxiter, float epsilon, float tau,
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

		iter_op_p_call(thresh, itrdata.scale * itrdata.tau, x, x);

		ravine(vops, N, &ra, x, o);	// FISTA
		iter_op_call(op, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		itrdata.rsnew = vops->norm(N, r);

		debug_printf(DP_DEBUG3, "#It %03d: %f   \n", itrdata.iter, itrdata.rsnew / itrdata.rsnot);

		if (itrdata.rsnew < epsilon)
			break;

		vops->axpy(N, x, itrdata.tau, r);
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
void landweber(unsigned int maxiter, float epsilon, float alpha, long N, long M,
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

	for (unsigned int i = 0; i < maxiter; i++) {

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
float conjgrad(unsigned int maxiter, float l2lambda, float epsilon, 
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


	unsigned int i = 0;

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
void irgnm(unsigned int iter, float alpha, float alpha_min, float redu, long N, long M,
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

	for (unsigned int i = 0; i < iter; i++) {

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
void irgnm2(unsigned int iter, float alpha, float alpha_min, float alpha_min0, float redu, long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	struct iter_op_s der,
	struct iter_op_p_s lsqr,
	float* x, const float* xref, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(M);
	float* q = vops->allocate(M);

	for (unsigned int i = 0; i < iter; i++) {

		iter_monitor(monitor, vops, x);

		iter_op_call(op, r, x);			// r = F x

		vops->xpay(M, -1., r, y);	// r = y - F x

		debug_printf(DP_DEBUG2, "Step: %u, Res: %f\n", i, vops->norm(M, r));

		if (NULL != xref)
			vops->axpy(N, x, -1., xref);

		iter_op_call(der, q, x);

		vops->axpy(M, r, +1., q);

		iter_op_p_call(lsqr, alpha, x, r);

		if (NULL != xref)
			vops->axpy(N, x, +1., xref);

		alpha = (alpha - alpha_min) / redu + alpha_min;

		if (alpha < alpha_min0)
			alpha = alpha_min0;

		if (NULL != callback.fun)
			iter_op_call(callback, x, x);
	}

	vops->del(q);
	vops->del(r);
}



/**
 * Alternating Minimzation
 *
 * Minimize residual by calling each min_op in turn.
 */
void altmin(unsigned int iter, float alpha, float redu,
	    long N,
	    const struct vec_iter_s* vops,
	    unsigned int NI,
	    struct iter_nlop_s op,
	    struct iter_op_p_s min_ops[__VLA(NI)],
	    float* x[__VLA(NI)], const float* y,
	    struct iter_nlop_s callback)
{
	float* r = vops->allocate(N);
	vops->clear(N, r);


	float* args[1 + NI];
	args[0] = r;

	for (long i = 0; i < NI; ++i)
		args[1 + i] = x[i];

	for (unsigned int i = 0; i < iter; i++) {

		for (unsigned int j = 0; j < NI; ++j) {

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
void pocs(unsigned int maxiter,
	unsigned int D, struct iter_op_p_s proj_ops[static D],
	const struct vec_iter_s* vops,
	long N, float* x,
	struct iter_monitor_s* monitor)
{
	UNUSED(N);
	UNUSED(vops);

	for (unsigned int i = 0; i < maxiter; i++) {

		debug_printf(DP_DEBUG3, "#Iter %d\n", i);

		iter_monitor(monitor, vops, x);

		for (unsigned int j = 0; j < D; j++)
			iter_op_p_call(proj_ops[j], 1., x, x); // use temporary memory here?
	}
}


/**
 *  Power iteration
 */
double power(unsigned int maxiter,
	long N,
	const struct vec_iter_s* vops,
	struct iter_op_s op,
	float* u)
{
	double s = vops->norm(N, u);
	vops->smul(N, 1. / s, u, u);

	for (unsigned int i = 0; i < maxiter; i++) {

		iter_op_call(op, u, u);		// r = A x

		s = vops->norm(N, u);
		vops->smul(N, 1. / s, u, u);
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
 * @param decay decay rate
 * @param theta convex combination rate
 * @param N size of input, x
 * @param M size of transformed input, Ax
 * @param vops vector ops definition
 * @param op_forw forward operator, A
 * @param op_adj adjoint operator, AH
 * @param prox1 proximal function of F, e.g. prox_l2ball
 * @param prox2 proximal function of G, e.g. prox_wavelet_thresh
 * @param x initial estimate
 * @param monitor callback function
 */
void chambolle_pock(unsigned int maxiter, float epsilon, float tau, float sigma, float theta, float decay,
	long N, long M,
	const struct vec_iter_s* vops,
	struct iter_op_s op_forw,
	struct iter_op_s op_adj,
	struct iter_op_p_s prox1,
	struct iter_op_p_s prox2,
	float* x,
	struct iter_monitor_s* monitor)
{
	float* x_avg = vops->allocate(N);
	float* x_old = vops->allocate(N);
	float* x_new = vops->allocate(N);

	float* u_old = vops->allocate(M);
	float* u = vops->allocate(M);
	float* u_new = vops->allocate(M);

	vops->copy(N, x_old, x);
	vops->copy(N, x_new, x);
	vops->copy(N, x_avg, x);

	vops->clear(M, u);
	vops->clear(M, u_new);
	vops->clear(M, u_old);


	for (unsigned int i = 0; i < maxiter; i++) {

		float lambda = (float)pow(decay, i);

		/* update u
		 * u0 = u
		 * p = u + sigma * A(x)
		 * u = p - sigma * prox1(p / sigma, 1 / sigma)
		 * u = lambda * u + (1 - lambda) * u0
		 */

		iter_op_call(op_forw, u_old, x_avg);

		vops->axpy(M, u_old, 1. / sigma, u); // (u + sigma * A(x)) / sigma

		iter_op_p_call(prox1, 1. / sigma, u_new, u_old);

		vops->axpbz(M, u_new, -1. * sigma, u_new, sigma, u_old);
		vops->copy(M, u_old, u);
		vops->axpbz(M, u, lambda, u_new, 1. - lambda, u_old);

		/* update x
		 * x0 = x
		 * q = x0 - tau * AH(u)
		 * x = prox2(q, tau)
		 * x = lambda * x + (1 - lambda * x0)
		 */
		vops->copy(N, x_old, x);

		iter_op_call(op_adj, x_new, u);

		vops->axpy(N, x, -1. * tau, x_new);

		iter_op_p_call(prox2, tau, x_new, x);

		vops->axpbz(N, x, lambda, x_new, 1. - lambda, x_old);

		/* update x_avg
		 * a_avg = x + theta * (x - x0)
		 */
		vops->axpbz(N, x_avg, 1 + theta, x, -1. * theta, x_old);

		// residual
		vops->sub(N, x_old, x, x_old);
		vops->sub(M, u_old, u, u_old);

		float res1 = vops->norm(N, x_old) / sigma;
		float res2 = vops->norm(M, u_old) / tau;

		iter_monitor(monitor, vops, x);

		debug_printf(DP_DEBUG3, "#It %03d: %f %f  \n", i, res1, res2);

		if (epsilon > (res1 + res2))
			break;
	}

	debug_printf(DP_DEBUG3, "\n");

	vops->del(x_avg);
	vops->del(x_old);
	vops->del(x_new);

	vops->del(u_old);
	vops->del(u);
	vops->del(u_new);
}

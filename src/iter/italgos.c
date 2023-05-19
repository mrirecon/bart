/* Copyright 2013-2017. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Center Göttingen.
 * Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2023 Martin Uecker <uecker@tugraz.at>
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

extern inline void iter_op_call(struct iter_op_s op, float* dst, const float* src);
extern inline void iter_nlop_call(struct iter_nlop_s op, int N, float* args[N]);
extern inline void iter_nlop_call_select_der(struct iter_nlop_s op, int N, float* args[N], unsigned long der_out, unsigned long der_in);
extern inline void iter_op_p_call(struct iter_op_p_s op, float rho, float* dst, const float* src);
extern inline void iter_op_arr_call(struct iter_op_arr_s op, int NO, unsigned long oflags, float* dst[NO], int NI, unsigned long iflags, const float* src[NI]);

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









void landweber_sym(int maxiter, float epsilon, float alpha, long N,
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
void fista(int maxiter, float epsilon, float tau,
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
	struct iter_op_p_s lsqr,
	float* x, const float* xref, const float* y,
	struct iter_op_s callback,
	struct iter_monitor_s* monitor)
{
	float* r = vops->allocate(M);
	float* q = vops->allocate(M);

	for (int i = 0; i < iter; i++) {

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
	long N, float* x,
	struct iter_monitor_s* monitor)
{
	UNUSED(N);
	UNUSED(vops);

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
	float* u)
{
	double s = vops->norm(N, u);
	vops->smul(N, 1. / s, u, u);

	for (int i = 0; i < maxiter; i++) {

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
void chambolle_pock(int maxiter, float epsilon, float tau, float sigma, float theta, float decay,
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

	float* u = vops->allocate(M);

	vops->copy(N, x_avg, x);

	vops->clear(M, u);

	for (int i = 0; i < maxiter; i++) {

		float lambda = (float)pow(decay, i);

		/* update u
		 * u0 = u
		 * p = u + sigma * A(x)
		 * u = p - sigma * prox1(p / sigma, 1 / sigma)
		 * u = lambda * u + (1 - lambda) * u0
		 */

		float* u_old = vops->allocate(M);
		float* u_new = vops->allocate(M);

		iter_op_call(op_forw, u_old, x_avg);

		vops->axpy(M, u_old, 1. / sigma, u); // (u + sigma * A(x)) / sigma

		iter_op_p_call(prox1, 1. / sigma, u_new, u_old);

		vops->axpbz(M, u_new, -1. * sigma, u_new, sigma, u_old);
		vops->copy(M, u_old, u);
		vops->axpbz(M, u, lambda, u_new, 1. - lambda, u_old);

		vops->sub(M, u_old, u, u_old);
		float res2 = vops->norm(M, u_old) / tau;

		vops->del(u_old);
		vops->del(u_new);

		/* update x
		 * x0 = x
		 * q = x0 - tau * AH(u)
		 * x = prox2(q, tau)
		 * x = lambda * x + (1 - lambda * x0)
		 */

		float* x_old = vops->allocate(N);
		float* x_new = vops->allocate(N);

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
	vops->del(u);
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
static float compute_objective(long NO, long NI, struct iter_nlop_s nlop, float* args[NO + NI], unsigned long out_optimize_flag, unsigned long der_in_flag, const struct vec_iter_s* vops)
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
		float learning_rate_schedule[epochs][batches],
		long NI, long isize[NI], enum IN_TYPE in_type[NI], float* x[NI],
		long NO, long osize[NO], enum OUT_TYPE out_type[NI],
		int N_batch, int N_total,
		const struct vec_iter_s* vops,
		struct iter_nlop_s nlop, struct iter_op_arr_s adj,
		struct iter_op_p_s update[NI],
		struct iter_op_p_s prox[NI],
		struct iter_nlop_s nlop_batch_gen,
		struct iter_op_s callback, struct monitor_iter6_s* monitor, const struct iter_dump_s* dump)
{
	UNUSED(callback);

	float* grad[NI];
	float* dxs[NI];
	float* args[NO + NI];

	float* x_batch_gen[NI]; //arrays which are filled by batch generator
	long N_batch_gen = 0;

	unsigned long in_optimize_flag = 0;
	unsigned long out_optimize_flag = 0;

	if ((int)batches != N_total / N_batch)
		error("Wrong number of batches!");

	for (int i = 0; i < NI; i++) {

		switch (in_type[i]) {

		case IN_STATIC:

			grad[i] = NULL;
			dxs[i] = NULL;
			break;

		case IN_BATCH:

			grad[i] = NULL;
			dxs[i] = NULL;
			break;

		case IN_OPTIMIZE:

			grad[i] = vops->allocate(isize[i]);
			dxs[i] = vops->allocate(isize[i]);

			in_optimize_flag = MD_SET(in_optimize_flag, i);

			if (NULL != prox[i].fun)
				iter_op_p_call(prox[i], 0, x[i], x[i]); //project to constraint

			break;

		case IN_BATCH_GENERATOR:

			grad[i] = NULL;
			dxs[i] = NULL;

			if (NULL != x[i])
				error("NULL != x[%d] for batch generator\n", i);

			x[i] = vops->allocate(isize[i]);

			x_batch_gen[N_batch_gen] = x[i];
			N_batch_gen += 1;
			break;

		case IN_BATCHNORM:

			grad[i] = NULL;
			dxs[i] = NULL;
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

	for (int epoch = 0; epoch < epochs; epoch++) {

		iter_dump(dump, epoch, NI, (const float**)x);

		for (int i_batch = 0; i_batch < N_total / N_batch; i_batch++) {

			if (0 != N_batch_gen)
				iter_nlop_call(nlop_batch_gen, N_batch_gen, x_batch_gen);

			float r0 = compute_objective(NO, NI, nlop, args, out_optimize_flag, in_optimize_flag, vops); // update graph and compute loss

			getgrad(NI, in_optimize_flag, isize, grad, NO, out_optimize_flag, adj, vops);

			int batchnorm_counter = 0;

			if (NULL != learning_rate_schedule)
				learning_rate = learning_rate_schedule[epoch][i_batch];

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

			monitor_iter6(monitor, epoch, i_batch, N_total / N_batch, r0, NI, (const float**)x, NULL);
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
void iPALM(	long NI, long isize[NI], enum IN_TYPE in_type[NI], float* x[NI], float* x_old[NI],
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
		struct iter_op_s callback, struct monitor_iter6_s* monitor, const struct iter_dump_s* dump)
{
	UNUSED(callback);

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

	for (int epoch = epoch_start; epoch < epoch_end; epoch++) {

		iter_dump(dump, epoch, NI, (const float**)x);

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
			char post_string[20 * NI];
			sprintf(post_string, " ");

			for (int i = 0; i < NI; i++)
				if (IN_OPTIMIZE == in_type[i])
					sprintf(post_string + strlen(post_string), "L[%d]=%.3e ", i, L[i]);

			monitor_iter6(monitor, epoch, batch, N_batch, r_i, NI, (const float**)x, post_string);
		}
	}


	for (int i = 0; i < NI; i++) {

		if (IN_BATCH_GENERATOR == in_type[i]) {

			vops->del(x[i]);
			x[i] = NULL;
		}
	}

	for (int o = 0; o < NO; o++)
		if(NULL != args[o])
			vops->del(args[o]);
}

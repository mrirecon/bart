/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013-2014 Frank Ong <frankong@berkeley.edu>
 * 2013-2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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
 */

#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#define NUM_INTERNAL
#include "num/vecops.h"
#include "num/multind.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "iter/vec.h"
#include "italgos.h"



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









void landweber_sym(unsigned int maxiter, float epsilon, float alpha, long N, void* data,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	float* x, const float* b)
{
	float* r = vops->allocate(N);

	double rsnot = vops->norm(N, b);

	for (unsigned int i = 0; i < maxiter; i++) {

		op(data, r, x);		// r = A x
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
 * Store information about iterative algorithm.
 * Used to flexibly modify behavior, e.g. continuation
 *
 * @param rsnew current residual
 * @param rsnot initial residual
 * @param iter current iteration
 * @param maxiter maximum iteration
 */
struct iter_data {

	double rsnew;
	double rsnot;
	unsigned int iter;
	const unsigned int maxiter;
};



/**
 * Continuation for regularization. Returns fraction to scale regularization parameter
 *
 * @param itrdata state of iterative algorithm
 * @param delta scaling of regularization in the final iteration (1. means don't scale, 0. means scale to zero)
 *
 */
static float ist_continuation(struct iter_data* itrdata, const float delta)
{
/*
	// for now, just divide into evenly spaced bins
	const float num_steps = itrdata->maxiter - 1;

	int step = (int)(itrdata->iter * num_steps / (itrdata->maxiter - 1));

	float scale = 1. - (1. - delta) * step / num_steps;

	return scale;
*/
	float a = logf( delta ) / (float) itrdata->maxiter;
	return expf( a * itrdata->iter );
}



/**
 * Iterative Soft Thresholding
 *
 * @param maxiter maximum number of iterations
 * @param epsilon stop criterion
 * @param tau (step size) weighting on the residual term, A^H (b - Ax)
 * @param lambda_start initial regularization weighting
 * @param lambda_end final regularization weighting (for continuation)
 * @param N size of input, x
 * @param data structure, e.g. sense_data
 * @param vops vector ops definition
 * @param op linear operator, e.g. A
 * @param thresh threshold function, e.g. complex soft threshold
 * @param x initial estimate
 * @param b observations
 */
void ist(unsigned int maxiter, float epsilon, float tau,
		float continuation, bool hogwild, long N, void* data,
		const struct vec_iter_s* vops,
		void (*op)(void* data, float* dst, const float* src), 
		void (*thresh)(void* data, float lambda, float* dst, const float* src),
		void* tdata,
		float* x, const float* b, const float* x_truth,
		void* obj_eval_data,
		float (*obj_eval)(const void*, const float*))
{

	struct iter_data itrdata = {

		.rsnew = 1.,
		.rsnot = 1.,
		.iter = 0,
		.maxiter = maxiter,
	};

	float* r = vops->allocate(N);

	float* x_err = NULL;
	if (NULL != x_truth)
		x_err = vops->allocate(N);

	itrdata.rsnot = vops->norm(N, b);

	float ls_old = 1.;
	float lambda_scale = 1.;

	int hogwild_k = 0;
	int hogwild_K = 10;


	for (itrdata.iter = 0; itrdata.iter < maxiter; itrdata.iter++) {

		if (NULL != x_truth) {

			vops->sub(N, x_err, x, x_truth);
			debug_printf(DP_DEBUG3, "relMSE = %f\n", vops->norm(N, x_err) / vops->norm(N, x_truth));
		}

		if (NULL != obj_eval) {

			float objval = obj_eval(obj_eval_data, x);
			debug_printf(DP_DEBUG3, "#%d OBJVAL= %f\n", itrdata.iter, objval);
		}

		ls_old = lambda_scale;
		lambda_scale = ist_continuation(&itrdata, continuation);
		
		if (lambda_scale != ls_old) 
			debug_printf(DP_DEBUG3, "##lambda_scale = %f\n", lambda_scale);


		thresh(tdata, tau, x, x);


		op(data, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		itrdata.rsnew = vops->norm(N, r);

		debug_printf(DP_DEBUG3, "#It %03d: %f \n", itrdata.iter, itrdata.rsnew / itrdata.rsnot);

		if (itrdata.rsnew < epsilon)
			break;

		vops->axpy(N, x, tau * lambda_scale, r);


		if (hogwild)
			hogwild_k++;
		
		if (hogwild_k == hogwild_K) {

			hogwild_K *= 2;
			hogwild_k = 0;
			tau /= 2;
		}

	}

	debug_printf(DP_DEBUG3, "\n");

	if (NULL != x_truth)
		vops->del(x_err);

	vops->del(r);
}



/**
 * Iterative Soft Thresholding/FISTA to solve min || b - Ax ||_2 + lambda || T x ||_1
 *
 * @param maxiter maximum number of iterations
 * @param epsilon stop criterion
 * @param tau (step size) weighting on the residual term, A^H (b - Ax)
 * @param lambda_start initial regularization weighting
 * @param lambda_end final regularization weighting (for continuation)
 * @param N size of input, x
 * @param data structure, e.g. sense_data
 * @param vops vector ops definition
 * @param op linear operator, e.g. A
 * @param thresh threshold function, e.g. complex soft threshold
 * @param x initial estimate
 * @param b observations
 */
void fista(unsigned int maxiter, float epsilon, float tau, 
	   float continuation, bool hogwild,
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   void (*op)(void* data, float* dst, const float* src), 
	   void (*thresh)(void* data, float lambda, float* dst, const float* src),
	   void* tdata,
	   float* x, const float* b, const float* x_truth,
	   void* obj_eval_data,
	   float (*obj_eval)(const void*, const float*))
{

	struct iter_data itrdata = {

		.rsnew = 1.,
		.rsnot = 1.,
		.iter = 0,
		.maxiter = maxiter,
	};

	float* r = vops->allocate(N);
	float* o = vops->allocate(N);

	float* x_err = NULL;

	if (NULL != x_truth)
		x_err = vops->allocate(N);

	float ra = 1.;
	vops->copy(N, o, x);

	itrdata.rsnot = vops->norm(N, b);

	float ls_old = 1.;
	float lambda_scale = 1.;

	int hogwild_k = 0;
	int hogwild_K = 10;

	for (itrdata.iter = 0; itrdata.iter < maxiter; itrdata.iter++) {

		if (NULL != x_truth) {

			vops->sub(N, x_err, x, x_truth);
			debug_printf(DP_DEBUG3, "relMSE = %f\n", vops->norm(N, x_err) / vops->norm(N, x_truth));
		}

		if (NULL != obj_eval) {

			float objval = obj_eval(obj_eval_data, x);
			debug_printf(DP_DEBUG3, "#%d OBJVAL= %f\n", itrdata.iter, objval);
		}

		ls_old = lambda_scale;
		lambda_scale = ist_continuation(&itrdata, continuation);
		
		if (lambda_scale != ls_old) 
			debug_printf(DP_DEBUG3, "##lambda_scale = %f\n", lambda_scale);


		thresh(tdata, lambda_scale * tau, x, x);

		ravine(vops, N, &ra, x, o);	// FISTA
		op(data, r, x);		// r = A x
		vops->xpay(N, -1., r, b);	// r = b - r = b - A x

		itrdata.rsnew = vops->norm(N, r);

		debug_printf(DP_DEBUG3, "#It %03d: %f   \n", itrdata.iter, itrdata.rsnew / itrdata.rsnot);

		if (itrdata.rsnew < epsilon)
			break;

		vops->axpy(N, x, tau, r);


		if (hogwild)
			hogwild_k++;
		
		if (hogwild_k == hogwild_K) {

			hogwild_K *= 2;
			hogwild_k = 0;
			tau /= 2;
		}
	}

	debug_printf(DP_DEBUG3, "\n");

	vops->del(o);
	vops->del(r);

	if (NULL != x_truth)
		vops->del(x_err);
}



/**
 *  Landweber L. An iteration formula for Fredholm integral equations of the
 *  first kind. Amer. J. Math. 1951; 73, 615-624.
 */
void landweber(unsigned int maxiter, float epsilon, float alpha, long N, long M, void* data,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	void (*adj)(void* data, float* dst, const float* src), 
	float* x, const float* b,
	float (*obj_eval)(const void*, const float*))
{
	float* r = vops->allocate(M);
	float* p = vops->allocate(N);

	double rsnot = vops->norm(M, b);

	UNUSED(obj_eval);

	for (unsigned int i = 0; i < maxiter; i++) {

		op(data, r, x);		// r = A x
		vops->xpay(M, -1., r, b);	// r = b - r = b - A x

		double rsnew = vops->norm(M, r);

		debug_printf(DP_DEBUG3, "#%d: %f\n", i, rsnew / rsnot);

		if (rsnew < epsilon)
			break;

		adj(data, p, r);
		vops->axpy(N, x, alpha, p);
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
 * @param data structure, e.g. sense_data
 * @param vops vector ops definition
 * @param linop linear operator, i.e. A
 * @param x initial estimate
 * @param b observations
 */
float conjgrad(unsigned int maxiter, float l2lambda, float epsilon, 
	long N, void* data,
	const struct vec_iter_s* vops,
	void (*linop)(void* data, float* dst, const float* src), 
	float* x, const float* b, const float* x_truth,
	void* obj_eval_data,
	float (*obj_eval)(const void*, const float*))
{
	float* r = vops->allocate(N);
	float* p = vops->allocate(N);
	float* Ap = vops->allocate(N);

	float* x_err = NULL;

	if (NULL != x_truth)
		x_err = vops->allocate(N);

	// The first calculation of the residual might not
	// be necessary in some cases...

	linop(data, r, x);		// r = A x
	vops->axpy(N, r, l2lambda, x);

	vops->xpay(N, -1., r, b);	// r = b - r = b - A x
	vops->copy(N, p, r);		// p = r

	float rsnot = (float)pow(vops->norm(N, r), 2.);
	float rsold = rsnot;
	float rsnew = rsnot;

	float eps_squared = pow(epsilon, 2.);

	if (0. == rsold) {

		debug_printf(DP_DEBUG3, "CG: early out\n");
		return 0.;
	}

	for (unsigned int i = 0; i < maxiter; i++) {
		
		if (NULL != x_truth) {

			vops->sub(N, x_err, x, x_truth);
			debug_printf(DP_DEBUG2, "relMSE = %f\n", vops->norm(N, x_err) / vops->norm(N, x_truth));
		}

		if ((NULL != obj_eval) && (NULL != obj_eval_data)) {

			float objval = obj_eval(obj_eval_data, x);
			debug_printf(DP_DEBUG2, "#CG%d OBJVAL= %f\n", i, objval);
		}

		debug_printf(DP_DEBUG3, "#%d: %f\n", i, (double)sqrtf(rsnew));

		linop(data, Ap, p);	// Ap = A p
		vops->axpy(N, Ap, l2lambda, p);

		float pAp = (float)vops->dot(N, p, Ap);

		if (0. == pAp)
			break;

		float alpha = rsold / pAp;

		vops->axpy(N, x, +alpha, p);
		vops->axpy(N, r, -alpha, Ap);
	
		rsnew = (float)pow(vops->norm(N, r), 2.);
		float beta = rsnew / rsold;
		
		rsold = rsnew;

		if (rsnew <= eps_squared) {
			//debug_printf(DP_DEBUG3, "%d ", i);
			break;
		}

		vops->xpay(N, beta, p, r);	// p = beta * p + r

	}

	vops->del(Ap);
	vops->del(p);
	vops->del(r);

	if (NULL != x_truth)
		vops->del(x_err);

	return sqrtf(rsnew);
}



/**
 * Conjugate Gradient Descent with history saving.
 * The history should be preallocated
 */
float conjgrad_hist(struct iter_history_s* history, unsigned int maxiter, float l2lambda, float epsilon, 
	long N, void* data,
	const struct vec_iter_s* vops,
	void (*linop)(void* data, float* dst, const float* src), 
	float* x, const float* b, const float* x_truth,
	void* obj_eval_data,
	float (*obj_eval)(const void*, const float*))
{
	float* r = vops->allocate(N);
	float* p = vops->allocate(N);
	float* Ap = vops->allocate(N);

	float* x_err = NULL;

	if (NULL != x_truth)
		x_err = vops->allocate(N);

	// The first calculation of the residual might not
	// be necessary in some cases...

	linop(data, r, x);		// r = A x
	vops->axpy(N, r, l2lambda, x);

	vops->xpay(N, -1., r, b);	// r = b - r = b - A x
	vops->copy(N, p, r);		// p = r

	float rsnot = (float)pow(vops->norm(N, r), 2.);
	float rsold = rsnot;
	float rsnew = rsnot;

	float eps_squared = pow(epsilon, 2.);

	history->numiter = 0;

	for (unsigned int i = 0; i < maxiter; i++) {

		history->numiter = i + 1;

		if (NULL != x_truth) {

			vops->sub(N, x_err, x, x_truth);
			float relMSE = vops->norm(N, x_err) / vops->norm(N, x_truth);
			history->relMSE[i] = relMSE;
			debug_printf(DP_DEBUG3, "relMSE = %f\n", relMSE);
		}

		if ((NULL != obj_eval) && (NULL != obj_eval_data)) {

			float objval = obj_eval(obj_eval_data, x);
			history->objective[i] = objval;
			debug_printf(DP_DEBUG3, "#CG%d OBJVAL= %f\n", i, objval);
		}

		//debug_printf(DP_DEBUG3, "#%d: %f\n", i, (double)sqrtf(rsnew));

		linop(data, Ap, p);	// Ap = A p
		vops->axpy(N, Ap, l2lambda, p);

		float alpha = rsold / (float)vops->dot(N, p, Ap);

		vops->axpy(N, x, +alpha, p);
		vops->axpy(N, r, -alpha, Ap);
	
		rsnew = (float)pow(vops->norm(N, r), 2.);
		float beta = rsnew / rsold;
		
		rsold = rsnew;

		if (rsnew <= eps_squared) {
			//debug_printf(DP_DEBUG3, "%d ", i);
			break;
		}

		vops->xpay(N, beta, p, r);	// p = beta * p + r

		history->resid[i] = sqrtf(rsnew);
	}


	// cleanup:
	vops->del(Ap);
	vops->del(p);
	vops->del(r);

	if (NULL != x_truth)
		vops->del(x_err);

	return sqrtf(rsnew);
}

/**
 * Internal variable used by conjgrad_hist_prealloc
 */
struct cg_data_s {

	float* r;
	float* p;
	float* Ap;
};

const struct cg_data_s* cg_data_init(long N, const struct vec_iter_s* vops) 
{
	PTR_ALLOC(struct cg_data_s, cgdata);
  
	cgdata->r = vops->allocate(N);
	cgdata->p = vops->allocate(N);
	cgdata->Ap = vops->allocate(N);

	return cgdata;
}

void cg_data_free(const struct cg_data_s* cgdata, const struct vec_iter_s* vops) 
{
	vops->del(cgdata->r);
	vops->del(cgdata->p);
	vops->del(cgdata->Ap);
}

/**
 * Conjugate Gradient Descent with history saving and pre-allocated data.
 * The history and cgdata should be preallocated.
 *
 * Preallocating the data is useful if the conjgrad is called within
 * an iteration loop (i.e. admm).
 */
float conjgrad_hist_prealloc(struct iter_history_s* history, unsigned int maxiter, float l2lambda, float epsilon, 
			     long N, void* data, struct cg_data_s* cgdata,
			     const struct vec_iter_s* vops,
			     void (*linop)(void* data, float* dst, const float* src), 
			     float* x, const float* b, const float* x_truth,
			     void* obj_eval_data,
			     float (*obj_eval)(const void*, const float*))
{
	float* r = cgdata->r;
	float* p = cgdata->p;
	float* Ap = cgdata->Ap;

	float* x_err = NULL;

	if (NULL != x_truth)
		x_err = vops->allocate(N);

	// The first calculation of the residual might not
	// be necessary in some cases...

	linop(data, r, x);		// r = A x
	vops->axpy(N, r, l2lambda, x);

	vops->xpay(N, -1., r, b);	// r = b - r = b - A x
	vops->copy(N, p, r);		// p = r

	float rsnot = (float)pow(vops->norm(N, r), 2.);
	float rsold = rsnot;
	float rsnew = rsnot;

	float eps_squared = pow(epsilon, 2.);


	for (unsigned int i = 0; i < maxiter; i++) {

		history->numiter = i + 1;
		
		if (NULL != x_truth) {

			vops->sub(N, x_err, x, x_truth);
			float relMSE = vops->norm(N, x_err) / vops->norm(N, x_truth);
			history->relMSE[i] = relMSE;
			debug_printf(DP_DEBUG3, "relMSE = %f\n", relMSE);
		}

		if ((NULL != obj_eval) && (NULL != obj_eval_data)) {

			float objval = obj_eval(obj_eval_data, x);
			history->objective[i] = objval;
			debug_printf(DP_DEBUG3, "#CG%d OBJVAL= %f\n", i, objval);
		}

		//debug_printf(DP_DEBUG3, "#%d: %f\n", i, (double)sqrtf(rsnew));

		linop(data, Ap, p);	// Ap = A p
		vops->axpy(N, Ap, l2lambda, p);

		float alpha = rsold / (float)vops->dot(N, p, Ap);

		vops->axpy(N, x, +alpha, p);
		vops->axpy(N, r, -alpha, Ap);
	
		rsnew = (float)pow(vops->norm(N, r), 2.);
		float beta = rsnew / rsold;
		
		rsold = rsnew;

		if (rsnew <= eps_squared) {
			//debug_printf(DP_DEBUG3, "%d ", i);
			break;
		}

		vops->xpay(N, beta, p, r);	// p = beta * p + r

		history->resid[i] = sqrtf(rsnew);
	}

	if (NULL != x_truth)
		vops->del(x_err);

	return sqrtf(rsnew);
}




/**
 * Iteratively Regularized Gauss-Newton Method
 * (Bakushinsky 1993)
 *
 * y = F(x) = F x0 + DF dx + ...
 *
 * IRGNM: DF^H ((y - F x_0) + DF (xn - x0)) = ( DF^H DF + alpha ) (dx + xn - x0)
 *        DF^H ((y - F x_0)) - alpha (xn - x0) = ( DF^H DF + alpha) dx
 */
void irgnm(unsigned int iter, float alpha, float redu, void* data, long N, long M,
	const struct vec_iter_s* vops,
	void (*op)(void* data, float* dst, const float* src), 
	void (*adj)(void* data, float* dst, const float* src), 
	void (*inv)(void* data, float alpha, float* dst, const float* src), 
	float* x, const float* x0, const float* y)
{
	float* r = vops->allocate(M);
	float* p = vops->allocate(N);
	float* h = vops->allocate(N);

	for (unsigned int i = 0; i < iter; i++) {

//		printf("#--------\n");

		op(data, r, x);			// r = F x

		vops->xpay(M, -1., r, y);	// r = y - F x

		debug_printf(DP_DEBUG3, "Res: %f\n", vops->norm(M, r));

		adj(data, p, r);	

		vops->axpy(N, p, +alpha, x0);
		vops->axpy(N, p, -alpha, x);

		inv(data, alpha, h, p);

		vops->axpy(N, x, 1., h);
		alpha /= redu;
	}

	vops->del(h);
	vops->del(p);
	vops->del(r);
}


/**
 * Projection onto Convex Sets
 *
 * minimize 0 subject to: x in C_1, x in C_2, ..., x in C_D,
 * where the C_i are convex sets
 */
void pocs(unsigned int maxiter,
	unsigned int D, const struct pocs_proj_op* proj_ops, 
	const struct vec_iter_s* vops,
	long N, float* x, const float* x_truth,
	void* obj_eval_data,
	float (*obj_eval)(const void*, const float*))
{

	float* x_err = NULL;

	if (NULL != x_truth)
		x_err = vops->allocate(N);

	for (unsigned int i = 0; i < maxiter; i++) {

		if (NULL != x_truth) {

			vops->sub(N, x_err, x, x_truth);
			debug_printf(DP_DEBUG3, "relMSE = %f\n", vops->norm(N, x_err) / vops->norm(N, x_truth));
		}

		for (unsigned int j = 0; j < D; j++)
			proj_ops[j].proj_fun(proj_ops[j].data, 1., x, x); // use temporary memory here?


		if (NULL != obj_eval) {

			float objval = obj_eval(obj_eval_data, x);
			debug_printf(DP_DEBUG3, "#%d OBJVAL= %f\n", i, objval);

		} else {

			debug_printf(DP_DEBUG3, "#Iter %d\n", i);
		}
	}

	if (NULL != x_truth)
		vops->del(x_err);
}


/**
 *  Power iteration
 */
double power(unsigned int maxiter,
	   long N, void* data,
	   const struct vec_iter_s* vops,
	   void (*op)(void* data, float* dst, const float* src), 
	   float* u)
{
	double s = vops->norm(N, u);
	vops->smul(N, 1. / s, u, u);

	for (unsigned int i = 0; i < maxiter; i++) {

		op(data, u, u);		// r = A x
		s = vops->norm(N, u);
		vops->smul(N, 1. / s, u, u);
	}

	return s;
}


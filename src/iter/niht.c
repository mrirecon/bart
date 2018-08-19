/* Copyright 2014-2016. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * Copyright 2017. University of Oxford.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Sofia Dimoudi <sofia.dimoudi@cardiov.ox.ac.uk>
 *
 *
 *
 * Blumensath T, Davies ME. Normalized iterative hard thresholding: Guaranteed 
 * stability and performance.
 * IEEE Journal of selected topics in signal processing. 2010;4:298-309.
 *
 * Blanchard JD, Tanner J. Performance comparisons of greedy algorithms in 
 * compressed sensing.
 * Numerical Linear Algebra with Applications. 2015;22:254-82.
 *
 */

#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <complex.h>

#include "num/ops.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/monitor.h"

#include "niht.h"

 
static void niht_imdom(const struct niht_conf_s* conf,  const struct vec_iter_s* vops,
	  struct iter_op_s op, struct iter_op_p_s thresh,
	  float* x, const float* b,
	  struct iter_monitor_s* monitor)
{
	double rsnew = 1.; // current residual
	double rsnot = 1.; // initial residual
	double rsold = 1.; // previous residual
  
	float mu = 1.; // step size
	unsigned int ic = 0; // iteration counter for criterion 3
	unsigned int iter = 0;
	long N = conf->N;
	
	float* r = vops->allocate(N);
	float* g = vops->allocate(N); // negative gradient of ||y - Ax||^2 with non-zero support
	float* m = vops->allocate(N); // non-zero support mask
	
	rsnot = vops->norm(N, b); // initial residual norm
	rsold = rsnot;

	// create initial support
	if (!conf->do_warmstart) {  //x_0 = 0, take support from b

		iter_op_p_call(thresh, 1.0, m, b);

		vops->zmul(N / 2, (complex float*)x, (complex float*)b, (complex float*)m);

	} else {  // x_0 has an initial value, take support from x

		iter_op_p_call(thresh, 1.0, m, x);

		vops->zmul(N / 2, (complex float*)x, (complex float*)x, (complex float*)m);
	}
	
	for (iter = 0; iter < conf->maxiter; iter++) {

		iter_monitor(monitor, vops, x);
    
		iter_op_call(op, r, x);   // r = A x

		vops->xpay(N, -1., r, b); // r = b - r = b - A x.

                // calculate step size.
		// 1. apply support x->g
		vops->zmul(N / 2, (complex float*)g, (complex float*)r, (complex float*)m);

		//mu = ||g_n||^2 / ||A g_n||^2
		double num = vops->dot(N, g, g);

		iter_op_call(op, g, g);

		double den = vops->dot(N, g, g);
		mu = num / den; //step size

		rsnew = vops->norm(N, r);
    
		debug_printf(DP_DEBUG3, "\n#It %03d relative residual r / r_0: %f \n", iter, rsnew / rsnot);
    
		// Stopping criteria: Blanchard and Tanner 2015
		// TODO: select appropriate epsilon and other criteria values
		if (rsnew < conf->epsilon) // residual is small
			break;
    
		if (rsnew > 100.0 * rsnot) // algorithm is diverging r_l > 100*r_0
			break;
    
		if (fabs(rsnew - rsold) <= 1.0E-06f) { // no significant change in residual

			debug_printf(DP_INFO, "\n*** rsnew - rsold =  %f **\n", fabs(rsnew - rsold) );
			ic++;

			if (15 == ic)        // in 16 iterations. Normally 1e-06
				break;             // more appropriate for noisy measurements
		}                      // where convergence will occur with larger residual
    
		vops->axpy(N, x, mu, r); // update solution: xk+1 = xk + mu rk+1

		iter_op_p_call(thresh, 1.0, m, x); // apply thresholding Hs(xk+1)

		vops->zmul(N / 2, (complex float*)x, (complex float*)x, (complex float*)m);
		
		rsold = rsnew; // keep residual for comparison
	}
  
	debug_printf(DP_DEBUG3, "\n");
  
	debug_printf(DP_DEBUG2, "\n#absolute residual: %f\n", rsnew);
	debug_printf(DP_DEBUG2, "\n#relative signal residual: %f\n\n", rsnew / rsnot);
  
	vops->del(r);
	vops->del(g);
	vops->del(m);
}


/**
 * Normalised Iterative Hard Thresholding/NIHT to solve min || b - Ax ||_2 s.t. || T x ||_0 <= k
 * using an adaptive step size with the iteration: x_n+1 = H_k (x_n + mu_n(A^T (y - A x_n))
 * where H_k(x) = support(x) the hard thresholding operator, keeps the k largest elements of x 
 * mu_n the adaptive step size.
 *
 * @param conf configuration params, eg maxiter, epsilon
 * @param trans linear transform operator, eg wavelets
 * @param vops vector ops definition
 * @param op linear operator, e.g. A
 * @param thresh NIHT threshold function
 * @param x initial estimate
 * @param b observations
 * @param monitor compute objective value, errors, etc.
 */
void niht(const struct niht_conf_s* conf, const struct niht_transop* trans, 
	  const struct vec_iter_s* vops,
	  struct iter_op_s op, struct iter_op_p_s thresh,
	  float* x, const float* b,
	  struct iter_monitor_s* monitor)
{
	if (0 == conf->trans) { // do NIHT in image domain

		niht_imdom(conf, vops, op, thresh, x, b, monitor);
		return;
	}
    
	double rsnew = 1.; // current residual
	double rsnot = 1.; // initial residual
	double rsold = 1.; // previous residual
  
	float mu = 1.; // step size
	unsigned int ic = 0; // iteration counter for criterion 3
	unsigned int iter = 0;
	long N = conf->N;
	long WN = trans->N;
	
	float* r = vops->allocate(N);
	float* g = vops->allocate(N); // negative gradient of ||y - Ax||^2 with non-zero support
	
	float* wg = vops->allocate(WN); // wavelet transform of g
	float* wx = vops->allocate(WN); // wavelet transform of x
	float* wm = vops->allocate(WN); // nonzero support mask with wavelet dimensions
	
	rsnot = vops->norm(N, b); // initial residual norm
	rsold = rsnot;
  
	// create initial support
	if (!conf->do_warmstart) {  //x_0 = 0, take support from b

		iter_op_call(trans->forward, wx, b);
		iter_op_p_call(thresh, 1.0, wm, wx); //produce mask by thresholding
		vops->zmul(WN / 2, (complex float*)wx, (complex float*)wx, (complex float*)wm); // apply mask

	} else { // x_0 has an initial value, take support from x

		iter_op_call(trans->forward, wx, x);
		iter_op_p_call(thresh, 1.0, wm, wx);
		vops->zmul(WN / 2, (complex float*)wx, (complex float*)wx, (complex float*)wm);
	}

	iter_op_call(trans->adjoint, x, wx);	
  
	for (iter = 0; iter < conf->maxiter; iter++) {

		iter_monitor(monitor, vops, x);
    
		iter_op_call(op, r, x);   // r = A x
		vops->xpay(N, -1., r, b); // r = b - r = b - A x.
    
		// calculate step size.
		// 1. apply support x->g
		iter_op_call(trans->forward, wg, r);

		vops->zmul(WN / 2, (complex float*)wg, (complex float*)wg, (complex float*)wm);

		iter_op_call(trans->adjoint, g, wg);

		// 2. mu = ||g_n||^2 / ||A g_n||^2
		double num = vops->dot(N, g, g);

		iter_op_call(op, g, g);		

		double den = vops->dot(N, g, g);

		mu = num / den; 
		debug_printf(DP_DEBUG3, "\n#step size: %f\n", mu);
		rsnew = vops->norm(N, r);
    
		debug_printf(DP_DEBUG3, "\n#It %03d relative residual r / r_0: %f \n", iter, rsnew / rsnot);
    
		// Stopping criteria: Blanchard and Tanner 2015
		if (rsnew < conf->epsilon) // residual is small
			break;

		if (rsnew > 100.0 * rsnot) // algorithm is diverging r_l > 100*r_0
			break;

		if (fabs(rsnew - rsold) <= 1.0E-06f) { // no significant change in residual

			debug_printf(DP_INFO, "\n*** rsnew - rsold =  %f **\n", fabs(rsnew - rsold));

			ic++;

			if (15 == ic)        // in 16 iterations. Normally 1e-06
				break;       // more appropriate for noisy measurements
		}                            // where convergence will occur with larger residual

		vops->axpy(N, x, mu, r); // update solution: xk+1 = xk + mu rk+1

		iter_op_call(trans->forward, wx, x);
		iter_op_p_call(thresh, 1.0, wm, wx); // apply thresholding Hs(xk+1)

		vops->zmul(WN / 2, (complex float*)wx, (complex float*)wx, (complex float*)wm);

		iter_op_call(trans->adjoint, x, wx);

		rsold = rsnew; // keep residual for comparison
	}

	debug_printf(DP_DEBUG3, "\n");

	debug_printf(DP_DEBUG2, "\n#absolute residual: %f\n", rsnew);
	debug_printf(DP_DEBUG2, "\n#relative signal residual: %f\n\n", rsnew / rsnot);

	vops->del(r);
	vops->del(g);
	vops->del(wg);
	vops->del(wx);
	vops->del(wm);
}


/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2011-12-13 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014-03-18 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * Afonso MA, Bioucas-Dias JM, Figueiredo M. An Augmented Lagrangian Approach to
 * the Constrained Optimization Formulation of Imaging Inverse Problems,
 * IEEE Trans Image Process, 20:681-695 (2011)
 *
 * Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. Distributed Optimization and
 * Statistical Learning via the Alternating Direction Method of Multipliers,
 * Foundations and Trends in Machine Learning, 3:1-122 (2011)
 *
 */
#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#define NUM_INTERNAL
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/vec.h"

#include "admm.h"



struct admm_normaleq_data {

	long N;
	unsigned int num_funs;
	struct admm_op* ops;
	float rho;
	void (*Aop)(void* _data, float* _dst, const float* _src);
	void* Aop_data;

        float* tmp;
};


static void admm_normaleq(void* _data, float* _dst, const float* _src)
{
	struct admm_normaleq_data* data = _data;
	long dims[1] = { data->N };

	//float* tmp = md_alloc_sameplace(1, dims, FL_SIZE, _src );

	md_clear(1, dims, _dst, sizeof(float));

	for (unsigned int i = 0; i < data->num_funs; i++) {

	        data->ops[i].normal(data->ops[i].data, data->tmp, _src);

		if ((NULL != data->Aop) && (NULL != data->Aop_data))
			md_axpy(1, dims, _dst, data->rho, data->tmp);
		else
			md_add(1, dims, _dst, _dst, data->tmp);
	}

	if ((NULL != data->Aop) && (NULL != data->Aop_data)) {

		data->Aop(data->Aop_data, data->tmp, _src);
		md_add(1, dims, _dst, _dst, data->tmp);
	}

	//md_free( tmp );
}


/*
 * ADMM (ADMM-2 from Afonso et al.)
 *
 * Solves min_x 0.5 || y - Ax ||_2^2 + sum_i f_i(G_i x), where the f_i are
 * arbitrary convex functions. If Aop is NULL, solves min_x sum_i f_i(G_i x)
 *
 * Each iteration requires solving the proximal of f_i, as well as applying
 * G_i, G_i^H, and G_i^H G_i, all which must be provided in admm_plan_s.
 */
void admm(struct admm_history_s* history, const struct admm_plan_s* plan, 
	  unsigned int D, const long z_dims[D],
	  long N, float* x, const float* x_adj,
	  const struct vec_iter_s* vops,
	  void (*Aop)(void* _data, float* _dst, const float* _src),
	  void* Aop_data,
	  void* obj_eval_data,
	  float (*obj_eval)(const void*, const float*))
{
	bool fast = plan->fast;
	double ABSTOL = plan->ABSTOL;
	double RELTOL = plan->RELTOL;
	float tau = plan->tau;
	float mu = plan->mu;


	unsigned int num_funs = D;

	long pos = 0;

	long fake_strs[num_funs];

	md_singleton_dims(num_funs, fake_strs);

	long M = md_calc_offset(num_funs, fake_strs, z_dims);

	// allocate memory for history
	history->r_norm = *TYPE_ALLOC(double[plan->maxiter]);
	history->s_norm = *TYPE_ALLOC(double[plan->maxiter]);
	history->eps_pri = *TYPE_ALLOC(double[plan->maxiter]);
	history->eps_dual = *TYPE_ALLOC(double[plan->maxiter]);
	history->objective = *TYPE_ALLOC(double[plan->maxiter]);
	history->rho = *TYPE_ALLOC(float[plan->maxiter]);
	history->relMSE = *TYPE_ALLOC(double[plan->maxiter]);

	long Mjmax = 0;
	for(unsigned int i = 0; i < num_funs; i++)
		Mjmax = MAX(Mjmax, z_dims[i]);

	struct iter_history_s cghistory;
	cghistory.numiter = 0;
	cghistory.relMSE = *TYPE_ALLOC(double[plan->maxitercg]);
	cghistory.objective = *TYPE_ALLOC(double[plan->maxitercg]);
	cghistory.resid = *TYPE_ALLOC(double[plan->maxitercg]);

	// allocate memory for all of our auxiliary variables
	float* z = vops->allocate(M);
	float* u = vops->allocate(M);
	float* rhs = vops->allocate(N);
	float* r = vops->allocate(M);
	float* s = vops->allocate(N);
	float* Gjx_plus_uj = vops->allocate(Mjmax);
	float* GH_usum = NULL;
	float* zj_old = NULL;


	if (!fast) {

		GH_usum = vops->allocate(N);
		zj_old = vops->allocate(Mjmax);
	}


	float* x_err = NULL;
	if (NULL != plan->image_truth)
		x_err = vops->allocate(N);

	if (!fast) {

		if (NULL != plan->image_truth)
			debug_printf(DP_DEBUG3, "%3s\t%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n", "iter", "cgiter", "rho", "r norm", "eps pri", "s norm", "eps dual", "obj", "relMSE");
		else
			debug_printf(DP_DEBUG3, "%3s\t%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n", "iter", "cgiter", "rho", "r norm", "eps pri", "s norm", "eps dual", "obj");
	}

	float rho = plan->rho;

	struct admm_normaleq_data ndata;
	ndata.N = N;
	ndata.num_funs = num_funs;
	ndata.ops = plan->ops;
	ndata.Aop = Aop;
	ndata.Aop_data = Aop_data;
	ndata.rho = 1.;
	ndata.tmp = vops->allocate(N);

	struct cg_data_s* cgdata = (struct cg_data_s*) cg_data_init(N, vops);

	// hogwild
	int hw_K = 1;
	int hw_k = 0;


	unsigned int grad_iter = 0; // keep track of number of gradient evaluations

	if (plan->do_warmstart) {

		for (unsigned int j = 0; j < num_funs; j++) {
	
			// initialize for j'th function update
			pos = md_calc_offset(j, fake_strs, z_dims);
			long Mj = z_dims[j];

			plan->ops[j].forward(plan->ops[j].data, Gjx_plus_uj, x); // Gj(x)

			if (0 == rho)
				vops->copy(Mj, z + pos, Gjx_plus_uj);
			else
				plan->prox_ops[j].prox_fun(plan->prox_ops[j].data, 1. / rho, z + pos, Gjx_plus_uj);

			vops->sub(Mj, u + pos, Gjx_plus_uj, z + pos);
		}

	} else {

		vops->clear(M, z);
		vops->clear(M, u);
	}


	for (unsigned int i = 0; i < plan->maxiter; i++) {

		// update x
		vops->clear(N, rhs);
		vops->sub(M, r, z, u);

		for (unsigned int j = 0; j < num_funs; j++) {

			pos = md_calc_offset(j, fake_strs, z_dims);
			plan->ops[j].adjoint(plan->ops[j].data, s, r + pos);
			vops->add(N, rhs, rhs, s);
		}


		if ((NULL != Aop) && (NULL != Aop_data)) {

			vops->xpay(N, rho, rhs, x_adj);
			ndata.rho = rho;
		}

		// x update: use plan->xupdate_fun if specified. use conjgrad otherwise
		if ((NULL != plan->xupdate_fun) && (NULL != plan->xupdate_data)) {

			plan->xupdate_fun(plan->xupdate_data, rho, x, rhs);
			grad_iter++;

		} else {

			float eps = vops->norm(N, rhs);

			if (eps > 0.) {

			  conjgrad_hist_prealloc(&cghistory, plan->maxitercg, 0., 1.E-3 * eps, N, &ndata, cgdata, vops, admm_normaleq, x, rhs, plan->image_truth, obj_eval_data, obj_eval);
			  //conjgrad_hist(&cghistory, plan->maxitercg, 0., 1.E-3 * eps, N, &ndata, vops, admm_normaleq, x, rhs, plan->image_truth, obj_eval_data, obj_eval);

			} else {

				cghistory.numiter = 0;
				cghistory.relMSE[0] = 0.;
				cghistory.objective[0] = 0.;
				cghistory.resid[0] = 0.;
			}

			grad_iter += cghistory.numiter;
		}

		if ((NULL != obj_eval) && (NULL != obj_eval_data))
			history->objective[i] = obj_eval(obj_eval_data, x);
		else
			history->objective[i] = 0.;



		double n1 = 0.;

		if (!fast) {

			vops->clear(N, GH_usum);
			vops->clear(N, s);
			vops->clear(M, r);
		}


		// z_j prox
		for (unsigned int j = 0; j < num_funs; j++) {
	
			// initialize for j'th function update
			pos = md_calc_offset(j, fake_strs, z_dims);
			long Mj = z_dims[j];


			plan->ops[j].forward(plan->ops[j].data, Gjx_plus_uj, x); // Gj(x)

			// over-relaxation: Gjx_hat = alpha * Gj(x) + (1 - alpha) * zj_old
			if (!fast) {

				vops->copy(Mj, zj_old, z + pos);
				vops->copy(Mj, r + pos, Gjx_plus_uj); // rj = Gj(x)
				n1 = n1 + vops->dot(Mj, r + pos, r + pos);

				vops->smul(Mj, plan->alpha, Gjx_plus_uj, Gjx_plus_uj);
				vops->axpy(Mj, Gjx_plus_uj, (1. - plan->alpha), z + pos);
			}

			vops->add(Mj, Gjx_plus_uj, Gjx_plus_uj, u + pos); // Gj(x) + uj

			if (0 == rho)
				vops->copy(Mj, z + pos, Gjx_plus_uj);
			else
				plan->prox_ops[j].prox_fun(plan->prox_ops[j].data, 1. / rho, z + pos, Gjx_plus_uj);

			vops->sub(Mj, u + pos, Gjx_plus_uj, z + pos);

			if (!fast) {

				// rj = rj - zj = Gj(x) - zj
				vops->sub(Mj, r + pos, r + pos, z + pos);

				// add next term to s: s = s + Gj^H (zj - zj_old)
				vops->sub(Mj, zj_old, z + pos, zj_old);
				plan->ops[j].adjoint(plan->ops[j].data, rhs, zj_old);
				vops->add(N, s, s, rhs);

				// GH_usum += G_j^H uj (for updating eps_dual)
				plan->ops[j].adjoint(plan->ops[j].data, rhs, u + pos);
				vops->add(N, GH_usum, GH_usum, rhs);
			}

		}

		history->rho[i] = rho;

		if (!fast) {

			history->s_norm[i] = rho * vops->norm(N, s); 

			history->r_norm[i] = vops->norm(M, r);

			n1 = sqrt(n1);
			double n2 = vops->norm(M, z);
			history->eps_pri[i] = ABSTOL * sqrt(M) + RELTOL * (n1 > n2 ? n1 : n2);

			history->eps_dual[i] = ABSTOL * sqrt(N) + RELTOL * rho * vops->norm(N, GH_usum);

			if (NULL != plan->image_truth) {

				vops->sub(N, x_err, x, plan->image_truth);
				history->relMSE[i] = vops->norm(N, x_err) / vops->norm(N, plan->image_truth);
			}


			if (NULL != plan->image_truth)
				debug_printf(DP_DEBUG3, "%3d\t%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n", i, grad_iter, history->rho[i], history->r_norm[i], history->eps_pri[i], history->s_norm[i], history->eps_dual[i], history->objective[i], history->relMSE[i]);
			else
				debug_printf(DP_DEBUG3, "%3d\t%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.5f\t%10.4f\n", i, grad_iter, history->rho[i], history->r_norm[i], history->eps_pri[i], history->s_norm[i], history->eps_dual[i], history->objective[i]);


			if ((grad_iter > plan->maxiter)
			    || ((history->r_norm[i] < history->eps_pri[i])
				&& (history->s_norm[i] < history->eps_dual[i]))) {

				history->numiter = i;
				break;
			}

			if (plan->dynamic_rho) {

				if (history->r_norm[i] > mu * history->s_norm[i]) {

					rho = rho * tau;
					vops->smul(M, 1. / tau, u, u);

				} else if (history->s_norm[i] > mu * history->r_norm[i]) {

					rho = rho / tau;
					vops->smul(M, tau, u, u);
				}
			}

		} else {

			debug_printf(DP_DEBUG3, "### ITER: %d (%d)\n", i, grad_iter);

			if (grad_iter > plan->maxiter)
				break;
		}

		if (plan->hogwild) {

			hw_k++;

			if (hw_k == hw_K) {

				hw_k = 0;
				rho *= 2.;
				hw_K *= 2;
				vops->smul(M, 0.5, u, u);
			}
		}

	}


	// cleanup
	vops->del(z);
	vops->del(u);
	vops->del(rhs);
	vops->del(Gjx_plus_uj);
	vops->del(r);
	vops->del(s);

	if (!fast) {

		vops->del(GH_usum);
		vops->del(zj_old);
	}

	if (NULL != plan->image_truth)
		vops->del(x_err);

	vops->del(ndata.tmp);	
	cg_data_free(cgdata, vops);

	free(cghistory.resid);
	free(cghistory.objective);
	free(cghistory.relMSE);

	free(history->r_norm);
	free(history->s_norm);
	free(history->eps_pri);
	free(history->eps_dual);
	free(history->objective);
	free(history->rho);
}

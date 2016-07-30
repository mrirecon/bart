/* Copyright 2014-2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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
#include <stdbool.h>

#include "num/ops.h"

#include "misc/debug.h"
#include "misc/misc.h"

#include "iter/italgos.h"
#include "iter/vec.h"

#include "admm.h"



struct admm_normaleq_data {

	long N;
	unsigned int num_funs;
	struct admm_op* ops;

	float rho;

	const struct vec_iter_s* vops;

	void (*Aop)(void* _data, float* _dst, const float* _src);
	void* Aop_data;

	float* tmp;
};


static void admm_normaleq(void* _data, float* _dst, const float* _src)
{
	struct admm_normaleq_data* data = _data;

	//float* tmp = alloc(data->N);

	data->vops->clear(data->N, _dst);

	for (unsigned int i = 0; i < data->num_funs; i++) {

	        data->ops[i].normal(data->ops[i].data, data->tmp, _src);

		if (NULL != data->Aop)
			data->vops->axpy(data->N, _dst, data->rho, data->tmp);
		else
			data->vops->add(data->N, _dst, _dst, data->tmp);
	}

	if (NULL != data->Aop) {

		data->Aop(data->Aop_data, data->tmp, _src);
		data->vops->add(data->N, _dst, _dst, data->tmp);
	}

	// del(tmp);
}



static long sum_long_array(unsigned int N, const long a[N])
{
	return ((0 == N) ? 0 : (a[0] + sum_long_array(N - 1, a + 1)));
}


/*
 * ADMM (ADMM-2 from Afonso et al.)
 *
 * Solves min_x 0.5 || y - Ax ||_2^2 + sum_i f_i(G_i x - b_i), where the f_i are
 * arbitrary convex functions. If Aop is NULL, solves min_x sum_i f_i(G_i x - b_i)
 *
 * Each iteration requires solving the proximal of f_i, as well as applying
 * G_i, G_i^H, and G_i^H G_i, all which must be provided in admm_plan_s.
 * The b_i are offsets (biases) that should also be provided in admm_plan_s.
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
	long M = sum_long_array(num_funs, z_dims);

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

	struct iter_history_s cghistory = {

		.numiter = 0,
		.relMSE = *TYPE_ALLOC(double[plan->maxitercg]),
		.objective = *TYPE_ALLOC(double[plan->maxitercg]),
		.resid = *TYPE_ALLOC(double[plan->maxitercg]),
	};

	// allocate memory for all of our auxiliary variables
	float* z_all = vops->allocate(M);
	float* u_all = vops->allocate(M);
	float* rhs = vops->allocate(N);
	float* r_all = vops->allocate(M);
	float* s = vops->allocate(N);
	float* Gjx_plus_uj = vops->allocate(Mjmax);
	float* GH_usum = NULL;
	float* zj_old = NULL;


	float* z[num_funs];
	float* u[num_funs];
	float* r[num_funs];

	for (unsigned int j = 0; j < num_funs; j++) {

		pos = sum_long_array(j, z_dims);

		z[j] = z_all + pos;
		u[j] = u_all + pos;
		r[j] = r_all + pos;
	}

	if (!fast) {

		GH_usum = vops->allocate(N);
		zj_old = vops->allocate(Mjmax);
	}


	float* x_err = NULL;
	float image_truth_norm = 0.;

	if (NULL != plan->image_truth) {

		image_truth_norm = vops->norm(N, plan->image_truth);

		x_err = vops->allocate(N);
	}

	if (!fast) {

		debug_printf(DP_DEBUG2, "%3s\t%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t", "iter", "cgiter", "rho", "r norm", "eps pri", "s norm", "eps dual", "obj");

		if (NULL != plan->image_truth)
			debug_printf(DP_DEBUG2, "%10s", "relMSE");

		debug_printf(DP_DEBUG2, "\n");
	}

	float rho = plan->rho;

	struct admm_normaleq_data ndata = {

		.N = N,
		.num_funs = num_funs,
		.ops = plan->ops,
		.Aop = Aop,
		.Aop_data = Aop_data,
		.rho = 1.,
		.tmp = vops->allocate(N),
		.vops = vops,
	};

	const struct cg_data_s* cgdata = cg_data_init(N, vops);

	// hogwild
	int hw_K = 1;
	int hw_k = 0;

	const float* biases[num_funs];

	for (unsigned int j = 0; j < num_funs; j++)
		biases[j] = (NULL != plan->biases) ? plan->biases[j] : NULL;

	// compute norm of biases -- for eps_primal
	double n3 = 0.;

	if (!fast) {

		for (unsigned int j = 0; j < num_funs; j++)
			if (biases[j] != NULL)
				n3 += vops->dot(z_dims[j], biases[j], biases[j]);

		n3 = sqrt(n3);
	}

	unsigned int grad_iter = 0; // keep track of number of gradient evaluations

	if (plan->do_warmstart) {

		for (unsigned int j = 0; j < num_funs; j++) {
	
			// initialize for j'th function update

			plan->ops[j].forward(plan->ops[j].data, Gjx_plus_uj, x); // Gj(x)

			if (NULL != biases[j])
				vops->sub(z_dims[j], Gjx_plus_uj, Gjx_plus_uj, biases[j]);

			if (0. == rho)
				vops->copy(z_dims[j], z[j], Gjx_plus_uj);
			else
				plan->prox_ops[j].prox_fun(plan->prox_ops[j].data, 1. / rho, z[j], Gjx_plus_uj);

			vops->sub(z_dims[j], u[j], Gjx_plus_uj, z[j]);
		}

	} else {

		vops->clear(M, z_all);
		vops->clear(M, u_all);
	}


	for (unsigned int i = 0; i < plan->maxiter; i++) {

		// update x
		vops->clear(N, rhs);
		vops->sub(M, r_all, z_all, u_all);

		for (unsigned int j = 0; j < num_funs; j++) {

			if (NULL != biases[j])
				vops->add(z_dims[j], r[j], r[j], biases[j]);

			plan->ops[j].adjoint(plan->ops[j].data, s, r[j]);
			vops->add(N, rhs, rhs, s);
		}

		if (NULL != Aop) {

			vops->xpay(N, rho, rhs, x_adj);
			ndata.rho = rho;
		}

		// x update: use plan->xupdate_fun if specified. use conjgrad otherwise
		if (NULL != plan->xupdate_fun) {

			plan->xupdate_fun(plan->xupdate_data, rho, x, rhs);
			grad_iter++;

		} else {

			float eps = vops->norm(N, rhs);

			if (eps > 0.) {

				conjgrad_hist_prealloc(&cghistory, plan->maxitercg, 0.,
						1.E-3 * eps, N, &ndata, cgdata, vops, admm_normaleq, x, rhs,
						plan->image_truth, obj_eval_data, obj_eval);

			  //conjgrad_hist(&cghistory, plan->maxitercg, 0., 1.E-3 * eps, N, &ndata, vops, admm_normaleq, x, rhs, plan->image_truth, obj_eval_data, obj_eval);

			} else {

				cghistory.numiter = 0;
				cghistory.relMSE[0] = 0.;
				cghistory.objective[0] = 0.;
				cghistory.resid[0] = 0.;
			}

			grad_iter += cghistory.numiter;
		}

		history->objective[i] = (NULL == obj_eval) ? 0. : obj_eval(obj_eval_data, x);


		double n1 = 0.;

		if (!fast) {

			vops->clear(N, GH_usum);
			vops->clear(N, s);
			vops->clear(M, r_all);
		}


		// z_j prox
		for (unsigned int j = 0; j < num_funs; j++) {
	
			// initialize for j'th function update

			plan->ops[j].forward(plan->ops[j].data, Gjx_plus_uj, x); // Gj(x)

			// over-relaxation: Gjx_hat = alpha * Gj(x) + (1 - alpha) * (zj_old + bj)
			if (!fast) {

				vops->copy(z_dims[j], zj_old, z[j]);
				vops->copy(z_dims[j], r[j], Gjx_plus_uj); // rj = Gj(x)

				n1 = n1 + vops->dot(z_dims[j], r[j], r[j]);

				vops->smul(z_dims[j], plan->alpha, Gjx_plus_uj, Gjx_plus_uj);
				vops->axpy(z_dims[j], Gjx_plus_uj, (1. - plan->alpha), z[j]);

				if (NULL != biases[j])
					vops->axpy(z_dims[j], Gjx_plus_uj, (1. - plan->alpha), biases[j]);
			}

			vops->add(z_dims[j], Gjx_plus_uj, Gjx_plus_uj, u[j]); // Gj(x) + uj

			if (NULL != biases[j])
				vops->sub(z_dims[j], Gjx_plus_uj, Gjx_plus_uj, biases[j]); // Gj(x) - bj + uj


			if (0. == rho)
				vops->copy(z_dims[j], z[j], Gjx_plus_uj);
			else
				plan->prox_ops[j].prox_fun(plan->prox_ops[j].data, 1. / rho, z[j], Gjx_plus_uj);

			vops->sub(z_dims[j], u[j], Gjx_plus_uj, z[j]);

			if (!fast) {

				// rj = rj - zj - bj = Gj(x) - zj - bj
				vops->sub(z_dims[j], r[j], r[j], z[j]);

				if (NULL != biases[j])
					vops->sub(z_dims[j], r[j], r[j], biases[j]);

				// add next term to s: s = s + Gj^H (zj - zj_old)
				vops->sub(z_dims[j], zj_old, z[j], z[j]);
				plan->ops[j].adjoint(plan->ops[j].data, rhs, zj_old);
				vops->add(N, s, s, rhs);

				// GH_usum += G_j^H uj (for updating eps_dual)
				plan->ops[j].adjoint(plan->ops[j].data, rhs, u[j]);
				vops->add(N, GH_usum, GH_usum, rhs);
			}

		}

		history->rho[i] = rho;

		if (!fast) {

			history->s_norm[i] = rho * vops->norm(N, s); 
			history->r_norm[i] = vops->norm(M, r_all);

			n1 = sqrt(n1);

			double n2 = vops->norm(M, z_all);
			double n = MAX(MAX(n1, n2), n3);

			history->eps_pri[i] = ABSTOL * sqrt(M) + RELTOL * n;
			history->eps_dual[i] = ABSTOL * sqrt(N) + RELTOL * rho * vops->norm(N, GH_usum);

			if (NULL != plan->image_truth) {

				vops->sub(N, x_err, x, plan->image_truth);
				history->relMSE[i] = vops->norm(N, x_err) / image_truth_norm;
			}

			debug_printf(DP_DEBUG2, "%3d\t%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.5f\t%10.4f",
						i, grad_iter, history->rho[i], history->r_norm[i],
						history->eps_pri[i], history->s_norm[i], history->eps_dual[i],
						history->objective[i]);

			if (NULL != plan->image_truth)
				debug_printf(DP_DEBUG2, "%10.4f", history->relMSE[i]);

			debug_printf(DP_DEBUG2, "\n");


			if (   (grad_iter > plan->maxiter)
			    || (   (history->r_norm[i] < history->eps_pri[i])
				&& (history->s_norm[i] < history->eps_dual[i]))) {

				history->numiter = i;
				break;
			}

			if (plan->dynamic_rho) {

				if (history->r_norm[i] > mu * history->s_norm[i]) {

					rho = rho * tau;
					vops->smul(M, 1. / tau, u_all, u_all);

				} else
				if (history->s_norm[i] > mu * history->r_norm[i]) {

					rho = rho / tau;
					vops->smul(M, tau, u_all, u_all);
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
				hw_K *= 2;
				rho *= 2.;

				vops->smul(M, 0.5, u_all, u_all);
			}
		}
	}


	// cleanup
	vops->del(z_all);
	vops->del(u_all);
	vops->del(rhs);
	vops->del(Gjx_plus_uj);
	vops->del(r_all);
	vops->del(s);

	if (!fast) {

		vops->del(GH_usum);
		vops->del(zj_old);
	}

	if (NULL != x_err)
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

/* Copyright 2014-2016. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014-2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 
 *
 *
 * Glowinski R and Marroco A. Sur l'approximation, par elements finis
 * d'ordre un, et la resolution, par penalisation-dualite d'une classe
 * de problemes de Dirichlet non lineaires. ESAIM: Mathematical
 * Modelling and Numerical Analysis - Modelisation Mathematique
 * et Analyse Numerique 9.R2: 41-76 (1975)
 *
 * Daniel Gabay and Bertrand Mercier.
 * A dual algorithm for the solution of nonlinear variational problems
 * via finite element approximation
 * Computers & Mathematics with Applications, 2:17-40 (1976)
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
#include <assert.h>

#include "num/ops.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/monitor.h"

#include "admm.h"


DEF_TYPEID(admm_history_s);






struct admm_normaleq_data {

	INTERFACE(iter_op_data);

	long N;
	unsigned int num_funs;
	struct admm_op* ops;

	float rho;

	const struct vec_iter_s* vops;

	unsigned int nr_invokes;

	struct iter_op_s Aop;
};

DEF_TYPEID(admm_normaleq_data);


static void admm_normaleq(iter_op_data* _data, float* dst, const float* src)
{
	struct admm_normaleq_data* data = CAST_DOWN(admm_normaleq_data, _data);

	float* tmp = data->vops->allocate(data->N);

	data->vops->clear(data->N, dst);

	for (unsigned int i = 0; i < data->num_funs; i++) {

	        iter_op_call(data->ops[i].normal, tmp, src);

		if (NULL != data->Aop.fun)
			data->vops->axpy(data->N, dst, data->rho, tmp);
		else
			data->vops->add(data->N, dst, dst, tmp);
	}

	data->nr_invokes++;

	if (NULL != data->Aop.fun) {

		iter_op_call(data->Aop, tmp, src);
		data->vops->add(data->N, dst, dst, tmp);
	}

	data->vops->del(tmp);
}


struct cg_xupdate_s {

	INTERFACE(iter_op_data);

	unsigned int N;
	const struct vec_iter_s* vops;

	unsigned int maxitercg;

	float cg_eps;

	struct admm_normaleq_data* ndata;

	struct iter_monitor_s* monitor;
};

DEF_TYPEID(cg_xupdate_s);

static void cg_xupdate(iter_op_data* _data, float rho, float* x, const float* rhs)
{
	struct cg_xupdate_s* data = CAST_DOWN(cg_xupdate_s, _data);
	assert(data->ndata->rho == rho);

	data->ndata->nr_invokes--;	// undo counting in admm

	float eps = data->vops->norm(data->N, rhs);

//	data->vops->clear(data->N, x);

	if (0. == eps)	// x should have been initialized already
		return;

	conjgrad(data->maxitercg, 0.,
			data->cg_eps * eps, data->N, data->vops,
			(struct iter_op_s){ admm_normaleq, CAST_UP(data->ndata) },
			x, rhs,
			data->monitor);

	data->ndata->nr_invokes--;	// subtract one for initialization in conjgrad
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
void admm(const struct admm_plan_s* plan,
	unsigned int D, const long z_dims[D],
	long N, float* x, const float* x_adj,
	const struct vec_iter_s* vops,
	struct iter_op_s Aop,
	struct iter_monitor_s* monitor)
{
	unsigned int num_funs = D;


	float* rhs = vops->allocate(N);
	float* s = vops->allocate(N);
	float* GH_usum = NULL;


	float* z[num_funs];
	float* u[num_funs];
	float* r[num_funs];

	for (unsigned int j = 0; j < num_funs; j++) {

		z[j] = vops->allocate(z_dims[j]);
		u[j] = vops->allocate(z_dims[j]);
		r[j] = vops->allocate(z_dims[j]);
	}


	if (!plan->fast)
		GH_usum = vops->allocate(N);

	float rho = plan->rho;

	struct admm_normaleq_data ndata = {

		.INTERFACE.TYPEID = &TYPEID(admm_normaleq_data),
		.N = N,
		.num_funs = num_funs,
		.ops = plan->ops,
		.Aop = Aop,
		.rho = 1.,
		.vops = vops,
		.nr_invokes = 0,
	};


	struct iter_op_p_s xupdate = plan->xupdate;

	struct cg_xupdate_s cg_xupdate_data = {

		.INTERFACE.TYPEID = &TYPEID(cg_xupdate_s),
		.N = N,
		.vops = vops,
		.maxitercg = plan->maxitercg,
		.cg_eps = plan->cg_eps,
		.ndata = &ndata,

		.monitor = monitor,
	};

	if (NULL == xupdate.fun)
		xupdate = (struct iter_op_p_s){ cg_xupdate, CAST_UP(&cg_xupdate_data) };


	// hogwild
	int hw_K = 1;
	int hw_k = 0;

	const float* biases[num_funs];

	for (unsigned int j = 0; j < num_funs; j++)
		biases[j] = (NULL != plan->biases) ? plan->biases[j] : NULL;

	// compute norm of biases -- for eps_primal
	double n3 = 0.;

	if (!plan->fast) {

		for (unsigned int j = 0; j < num_funs; j++)
			if (biases[j] != NULL)
				n3 += pow(vops->norm(z_dims[j], biases[j]), 2.);
	}


	if (plan->do_warmstart) {

		for (unsigned int j = 0; j < num_funs; j++) {
	
			// initialize for j'th function update

			float* Gjx_plus_uj = vops->allocate(z_dims[j]);

			iter_op_call(plan->ops[j].forward, Gjx_plus_uj, x); // Gj(x)

			if (NULL != biases[j])
				vops->sub(z_dims[j], Gjx_plus_uj, Gjx_plus_uj, biases[j]);

			if (0. == rho)
				vops->copy(z_dims[j], z[j], Gjx_plus_uj);
			else
				iter_op_p_call(plan->prox_ops[j], 1. / rho, z[j], Gjx_plus_uj);

			vops->sub(z_dims[j], u[j], Gjx_plus_uj, z[j]);

			vops->del(Gjx_plus_uj);
		}

	} else {

		for (unsigned int j = 0; j < num_funs; j++) {

			vops->clear(z_dims[j], z[j]);
			vops->clear(z_dims[j], u[j]);
		}
	}


	for (unsigned int i = 0; i < plan->maxiter; i++) {

		iter_monitor(monitor, vops, x);

		// update x
		vops->clear(N, rhs);

		for (unsigned int j = 0; j < num_funs; j++) {

			vops->sub(z_dims[j], r[j], z[j], u[j]);

			if (NULL != biases[j])
				vops->add(z_dims[j], r[j], r[j], biases[j]);

			iter_op_call(plan->ops[j].adjoint, s, r[j]);
			vops->add(N, rhs, rhs, s);
		}

		if (NULL != Aop.fun) {

			vops->xpay(N, rho, rhs, x_adj);
			ndata.rho = rho;
		}


		iter_op_p_call(xupdate, rho, x, rhs);
		ndata.nr_invokes++;


		double n1 = 0.;

		if (!plan->fast) {

			vops->clear(N, GH_usum);
			vops->clear(N, s);

			for (unsigned int j = 0; j < num_funs; j++)
				vops->clear(z_dims[j], r[j]);
		}


		// z_j prox
		for (unsigned int j = 0; j < num_funs; j++) {
	
			// initialize for j'th function update

			float* Gjx_plus_uj = vops->allocate(z_dims[j]);
			float* zj_old = vops->allocate(z_dims[j]);

			iter_op_call(plan->ops[j].forward, Gjx_plus_uj, x); // Gj(x)

			// over-relaxation: Gjx_hat = alpha * Gj(x) + (1 - alpha) * (zj_old + bj)
			if (!plan->fast) {

				vops->copy(z_dims[j], zj_old, z[j]);
				vops->copy(z_dims[j], r[j], Gjx_plus_uj); // rj = Gj(x)

				n1 += pow(vops->norm(z_dims[j], r[j]), 2.);

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
				iter_op_p_call(plan->prox_ops[j], 1. / rho, z[j], Gjx_plus_uj);

			vops->sub(z_dims[j], u[j], Gjx_plus_uj, z[j]);

			vops->del(Gjx_plus_uj);

			if (!plan->fast) {

				// rj = rj - zj - bj = Gj(x) - zj - bj
				vops->sub(z_dims[j], r[j], r[j], z[j]);

				if (NULL != biases[j])
					vops->sub(z_dims[j], r[j], r[j], biases[j]);

				// add next term to s: s = s + Gj^H (zj - zj_old)
				vops->sub(z_dims[j], zj_old, z[j], zj_old);
				iter_op_call(plan->ops[j].adjoint, rhs, zj_old);
				vops->add(N, s, s, rhs);

				// GH_usum += G_j^H uj (for updating eps_dual)
				iter_op_call(plan->ops[j].adjoint, rhs, u[j]);
				vops->add(N, GH_usum, GH_usum, rhs);
			}

			vops->del(zj_old);
		}

		float s_norm = 0.;
		float r_norm = 0.;

		if (plan->dynamic_rho || !plan->fast) {

			s_norm = rho * vops->norm(N, s);
			r_norm = 0.;

			for (unsigned int j = 0; j < num_funs; j++)
				r_norm += pow(vops->norm(z_dims[j], r[j]), 2.);

			r_norm = sqrt(r_norm);
		}

		if (!plan->fast) {

			double n2 = 0.;

			for (unsigned int j = 0; j < num_funs; j++)
				n2 += pow(vops->norm(z_dims[j], z[j]), 2.);

			double n = MAX(MAX(n1, n2), n3);

			long M = sum_long_array(num_funs, z_dims);

			float eps_pri = plan->ABSTOL * sqrt(M) + plan->RELTOL * sqrt(n);
			float eps_dual = plan->ABSTOL * sqrt(N) + plan->RELTOL * rho * vops->norm(N, GH_usum);


			struct admm_history_s history;

			history.s_norm = s_norm;
			history.r_norm = r_norm;
			history.eps_pri = eps_pri;
			history.eps_dual = eps_dual;
			history.rho = rho;
			history.numiter = i;
			history.nr_invokes = ndata.nr_invokes;

			iter_history(monitor, CAST_UP(&history));

			if (0 == i)
				debug_printf(DP_DEBUG2, "%3s\t%3s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\t%10s\n",
					"iter", "cgiter", "rho", "r norm", "eps pri",
					"s norm", "eps dual", "obj", "relMSE");


			debug_printf(DP_DEBUG2, "%3d\t%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\n",
				history.numiter, history.nr_invokes, history.rho,
				history.r_norm, history.eps_pri, history.s_norm, history.eps_dual,
				(NULL == monitor) ? -1. : monitor->obj,
				(NULL == monitor) ? -1. : monitor->err);


			if (   (ndata.nr_invokes > plan->maxiter)
			    || (   (r_norm < eps_pri)
				&& (s_norm < eps_dual)))
				break;

		} else {

			debug_printf(DP_DEBUG3, "### ITER: %d (%d)\n", i, ndata.nr_invokes);

			if (ndata.nr_invokes > plan->maxiter)
				break;
		}

		float sc = 1.;

		assert(!(plan->dynamic_rho && plan->hogwild));

		if (plan->dynamic_rho) {

			if (r_norm > plan->mu * s_norm)
				sc = plan->tau;
			else
			if (s_norm > plan->mu * r_norm)
				sc = 1. / plan->tau;
		}

		if (plan->hogwild) {

			hw_k++;

			if (hw_k == hw_K) {

				hw_k = 0;
				hw_K *= 2;
				sc = 2.;
			}
		}

		if (1. != sc) {

			rho = rho * sc;

			for (unsigned int j = 0; j < num_funs; j++)
				vops->smul(z_dims[j], 1. / sc, u[j], u[j]);
		}
	}


	// cleanup
	vops->del(rhs);
	vops->del(s);

	if (!plan->fast)
		vops->del(GH_usum);
}

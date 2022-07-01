/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <assert.h>
#include <math.h>

#include "num/ops.h"
#include "num/ops_p.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "nlops/nlop.h"

#include "misc/misc.h"
#include "misc/types.h"

#include "iter/batch_gen.h"
#include "iter/italgos.h"
#include "iter/vec.h"
#include "iter/iter2.h"
#include "iter/iter6_ops.h"
#include "iter/monitor_iter6.h"
#include "iter/iter_dump.h"

#include "iter/prox.h"

#include "iter6.h"

#ifndef STRUCT_TMP_COPY
#define STRUCT_TMP_COPY(x) ({ __typeof(x) __foo = (x); __typeof(__foo)* __foo2 = alloca(sizeof(__foo)); *__foo2 = __foo; __foo2; })
#endif
#define NLOP2ITNLOP(nlop) (struct iter_nlop_s){ (NULL == nlop) ? NULL : iter6_nlop, CAST_UP(STRUCT_TMP_COPY(((struct iter6_nlop_s){ { &TYPEID(iter6_nlop_s) }, nlop }))) }
#define NLOP2IT_ADJ_ARR(nlop) ({\
	long NO = nlop_get_nr_out_args(nlop);\
	long NI = nlop_get_nr_in_args(nlop);\
	const struct operator_s** adj_ops = (const struct operator_s**) alloca(sizeof(struct operator_s*) * NI * NO);\
	for (int o = 0; o < NO; o++)\
		for (int i = 0; i < NI; i++)\
			adj_ops[i * NO + o] = nlop_get_derivative(nlop, o, i)->adjoint;\
	struct iter6_op_arr_s adj_ops_data = { { &TYPEID(iter6_op_arr_s) }, NI, NO, adj_ops};\
	(struct iter_op_arr_s){iter6_op_arr_fun_deradj, CAST_UP(STRUCT_TMP_COPY(adj_ops_data))} ;})


DEF_TYPEID(iter6_sgd_conf);
DEF_TYPEID(iter6_adadelta_conf);
DEF_TYPEID(iter6_adam_conf);
DEF_TYPEID(iter6_iPALM_conf);

#define ITER6_DEFAULT \
	.INTERFACE.epochs = 1, \
	.INTERFACE.clip_norm = 0., \
	.INTERFACE.clip_val = 0., \
	.INTERFACE.weight_decay = 0., \
	.INTERFACE.history_filename = NULL, \
	.INTERFACE.dump_filename = NULL, \
	.INTERFACE.dump_mod = -1, \
	.INTERFACE.batchnorm_momentum = .95, \
	.INTERFACE.batchgen_type = BATCH_GEN_SAME, \
	.INTERFACE.batch_seed = 123, \
	.INTERFACE.dump_flag = 0, \
	.INTERFACE.min_learning_rate = 0.,\
	.INTERFACE.epochs_warmup = 0.,\
	.INTERFACE.monitor_averaged_objective = false,\
	.INTERFACE.learning_rate_epoch_mod = 0,

const struct iter6_sgd_conf iter6_sgd_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_sgd_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 0.001,

	.momentum = 0.
};


const struct iter6_adadelta_conf iter6_adadelta_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adadelta_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 1.,

	.rho = 0.95
};

const struct iter6_adam_conf iter6_adam_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_adam_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = .001,

	.reset_epoch = -1,

	.epsilon = 1.e-7,

	.beta1 = 0.9,
	.beta2 = 0.999,
};


const struct iter6_iPALM_conf iter6_iPALM_conf_defaults = {

	.INTERFACE.TYPEID = &TYPEID2(iter6_iPALM_conf),

	ITER6_DEFAULT

	.INTERFACE.learning_rate = 1.,

	.Lmin = 1.e-10,
	.Lmax = 1.e10,
	.Lshrink = 1.2,
	.Lincrease = 2.,

	.alpha = -1.,
	.beta = -1.,
	.convex = false,

	.trivial_stepsize = false,

	.alpha_arr = NULL,
	.beta_arr =NULL,
	.convex_arr = NULL,

	.reduce_momentum = true,

};



struct iter6_nlop_s {

	INTERFACE(iter_op_data);

	const struct nlop_s* nlop;
};

DEF_TYPEID(iter6_nlop_s);

static void iter6_nlop(iter_op_data* _o, int N, float* args[N], unsigned long der_out, unsigned long der_in)
{
	const auto data = CAST_DOWN(iter6_nlop_s, _o);

	assert((unsigned int)N == operator_nr_args(data->nlop->op));

	nlop_generic_apply_select_derivative_unchecked(data->nlop, N, (void*)args, der_out, der_in);
}

struct iter6_op_arr_s {

	INTERFACE(iter_op_data);

	long NO;
	long NI;

	const struct operator_s** ops;
};

DEF_TYPEID(iter6_op_arr_s);

static void iter6_op_arr_fun_deradj(iter_op_data* _o, int NO, unsigned long oflags, float* dst[NO], int NI, unsigned long iflags, const float* src[NI])
{
	const auto data = CAST_DOWN(iter6_op_arr_s, _o);

	assert(NO == data->NO);
	assert(1 == NI);
	int i_index = -1;

	for (int i = 0; i < data->NI; i++) {

		if (MD_IS_SET(iflags, i)) {

			assert(-1 == i_index);
			i_index = i;
		}
	}

	assert(-1 != i_index);

	const struct operator_s* op_arr[NO];
	float* dst_t[NO];
	int NO_t = 0;

	for (int o = 0; o < NO; o++) {

		if (MD_IS_SET(oflags, o)) {

			op_arr[NO_t] = data->ops[o * data->NI + i_index];
			dst_t[NO_t] = dst[o];
			NO_t += 1;
		}
	}
#if 0
	for (int i = 0; i < NO_t; i++)
		operator_apply_unchecked(op_arr[i], ((complex float**)dst_t)[i], (const complex float*)(src[0]));
#else
	operator_apply_joined_unchecked(NO_t, op_arr, (complex float**)dst_t, (const complex float*)(src[0]));
#endif
}

static const struct iter_dump_s* iter6_dump_default_create(const char* base_filename, long save_mod, const struct nlop_s* nlop, unsigned long save_flag, long NI, enum IN_TYPE in_type[NI])
{
	int D[NI];
	const long* dims[NI];

	bool guess_save_flag = (0 == save_flag);

	for (int i = 0; i < NI; i++) {

		D[i] = nlop_generic_domain(nlop, i)->N;
		dims[i] = nlop_generic_domain(nlop, i)->dims;

		if ((guess_save_flag) && ((IN_OPTIMIZE == in_type[i]) || (IN_BATCHNORM == in_type[i])))
			save_flag = MD_SET(save_flag, i);
	}

	return iter_dump_default_create(base_filename, save_mod, NI, save_flag, D, dims);
}

static const struct operator_p_s* get_update_operator(const iter6_conf* conf, int N, const long dims[N], long numbatches)
{
	auto conf_adadelta = CAST_MAYBE(iter6_adadelta_conf, conf);

	if (NULL != conf_adadelta)
		return operator_adadelta_update_create(N, dims, conf_adadelta->rho, 1.e-7);

	auto conf_sgd = CAST_MAYBE(iter6_sgd_conf, conf);

	if (NULL != conf_sgd)
		return operator_sgd_update_create(N, dims);

	auto conf_adam = CAST_MAYBE(iter6_adam_conf, conf);

	if (NULL != conf_adam)
		return operator_adam_update_create(N, dims, conf_adam->beta1, conf_adam->beta2, conf_adam->epsilon, numbatches * conf_adam->reset_epoch);

	error("iter6_conf not SGD-like!\n");
	return NULL;
}

static const float* get_learning_rate_schedule_cosine_annealing(int epochs, int numbatches, float learning_rate, float min_learning_rate, int epoch_mod)
{
	long dims[2] = {numbatches, epochs};

	if (1 >= epoch_mod)
		return NULL;

	assert(0 <= min_learning_rate);

	float (*result)[numbatches] = (float (*)[numbatches])md_alloc(2, dims, FL_SIZE);

	for (int ie = 0; ie < dims[1]; ie++)
		for (int ib = 0; ib < dims[0]; ib++)
			result[ie][ib] = min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * (1 + cosf(M_PI * (float)(ie % epoch_mod) / (float)(epoch_mod-1)));

	return &(result[0][0]);
}

static const float* get_learning_rate_schedule_exponential_decay(int epochs, int numbatches, float learning_rate, float min_learning_rate)
{
	long dims[2] = {numbatches, epochs};

	if (0 >= min_learning_rate)
		return NULL;

	assert(0 <= min_learning_rate);

	float (*result)[numbatches] = (float (*)[numbatches])md_alloc(2, dims, FL_SIZE);

	for (int ie = 0; ie < dims[1]; ie++)
		for (int ib = 0; ib < dims[0]; ib++)
			result[ie][ib] = learning_rate * (expf(((float)ie) / (epochs - 1) * logf(min_learning_rate / learning_rate)));

	return &(result[0][0]);
}

static const float* learning_rate_schedule_add_warmup(int epochs, int numbatches, float learning_rate, int epochs_warmup, const float* schedule)
{
	if (0 == epochs_warmup)
		return schedule;
	
	long dims[2] = {numbatches, epochs};
	float (*result)[numbatches] = (float (*)[numbatches])md_alloc(2, dims, FL_SIZE);
	
	for (int ie = 0; ie < epochs_warmup; ie++)
		for (int ib = 0; ib < numbatches; ib++) {

			result[ie][ib] = learning_rate / (float)(epochs_warmup * numbatches) * (float)(ie * numbatches + ib);
		}
	
	for (int ie = 0; ie < epochs - epochs_warmup; ie++)
		for (int ib = 0; ib < numbatches; ib++) {

			result[ie + epochs_warmup][ib] = (NULL == schedule) ? learning_rate : schedule[numbatches * ie + ib];
		}
	
	if (NULL != schedule)
		md_free(schedule);
	
	return &(result[0][0]);
}



void iter6_sgd_like(	const iter6_conf* conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	struct iter_nlop_s nlop_iter = NLOP2ITNLOP(nlop);
	struct iter_op_arr_s adj_op_arr = NLOP2IT_ADJ_ARR(nlop);
	struct iter_nlop_s nlop_batch_gen_iter = NLOP2ITNLOP(nlop_batch_gen);

	const struct operator_p_s* prox_ops_weight_decay[NI];

	for (int i = 0; i < NI; i++) {

		if ((0 != conf->weight_decay) && (NULL == prox_ops[i]) && (IN_OPTIMIZE == in_type[i])) {

			prox_ops_weight_decay[i] = prox_leastsquares_create(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, conf->weight_decay, NULL);
			prox_ops[i] = prox_ops_weight_decay[i];

		} else {

			prox_ops_weight_decay[i] = NULL;
		}
	}

	struct iter_op_p_s prox_iter[NI];

	for (int i = 0; i < NI; i++)
		prox_iter[i] = OPERATOR_P2ITOP((NULL == prox_ops ? NULL : prox_ops[i]));

	long isize[NI];
	long osize[NO];

	//array of update operators
	const struct operator_p_s* upd_ops[NI];

	for (int i = 0; i < NI; i++) {

		upd_ops[i] = NULL;

		if (IN_OPTIMIZE != in_type[i])
			continue;

		upd_ops[i] = get_update_operator(conf, nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, numbatches);

		if ((0.0 != conf->clip_norm) || (0.0 != conf->clip_val)) {

			const struct operator_s* tmp1 = operator_clip_create(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, conf->clip_norm, conf->clip_val);
			const struct operator_p_s* tmp2 = upd_ops[i];
			upd_ops[i] = operator_p_pre_chain(tmp1, tmp2);
			operator_free(tmp1);
			operator_p_free(tmp2);
		}
	}

	struct iter_op_p_s upd_iter_ops[NI];

	for (unsigned int i = 0; i < NI; i++)
		upd_iter_ops[i] = OPERATOR_P2ITOP(upd_ops[i]);

	for (int i = 0; i < NI; i++)
		isize[i] = 2 * md_calc_size(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);

	for (int o = 0; o < NO; o++)
		osize[o] = 2 * md_calc_size(nlop_generic_codomain(nlop, o)->N, nlop_generic_codomain(nlop, o)->dims);

	//gpu ref (dst[i] can be null if batch_gen)
	float* gpu_ref = NULL;
	for (int i = 0; i < NI; i++)
		if (IN_OPTIMIZE == in_type[i])
			gpu_ref = dst[i];

	assert(NULL != gpu_ref);

	bool free_monitor = (NULL == monitor);

	if (free_monitor)
		monitor = monitor_iter6_create(true, false, 0, NULL);

	if (conf->monitor_averaged_objective)
		monitor6_average_objective(monitor);


	const struct iter_dump_s* dump = NULL;

	if (   (NULL != conf->dump_filename)
	    && (0 < conf->dump_mod))
		dump = iter6_dump_default_create(conf->dump_filename, conf->dump_mod, nlop, conf->dump_flag, NI, in_type);

	float (*learning_rate_schedule)[numbatches] = NULL;
	
	if (0 < conf->min_learning_rate) {

		if (conf->learning_rate_epoch_mod)
			learning_rate_schedule = (float (*)[numbatches])get_learning_rate_schedule_cosine_annealing(conf->epochs, numbatches, conf->learning_rate, conf->min_learning_rate, conf->learning_rate_epoch_mod);
		else
			learning_rate_schedule = (float (*)[numbatches])get_learning_rate_schedule_exponential_decay(conf->epochs, numbatches, conf->learning_rate, conf->min_learning_rate);
	}

	learning_rate_schedule = (float (*)[numbatches])learning_rate_schedule_add_warmup(conf->epochs, numbatches, conf->learning_rate, conf->epochs_warmup, (const float*)learning_rate_schedule);

	sgd(	conf->epochs, numbatches,
		conf->learning_rate, conf->batchnorm_momentum,
		learning_rate_schedule,
		NI, isize, in_type, dst,
		NO, osize, out_type,
		batchsize, batchsize * numbatches,
		select_vecops(gpu_ref),
		nlop_iter, adj_op_arr,
		upd_iter_ops,
		prox_iter,
		nlop_batch_gen_iter,
		(struct iter_op_s){ NULL, NULL }, monitor, dump);

	for (int i = 0; i < NI; i++)
		operator_p_free(upd_ops[i]);

	if (NULL != conf->history_filename)
		monitor_iter6_dump_record(monitor, conf->history_filename);

	if (free_monitor)
		monitor_iter6_free(monitor);

	if (NULL != dump)
		iter_dump_free(dump);

	for (int i = 0; i < NI; i++) {

		if (NULL != prox_ops_weight_decay[i]) {

			operator_p_free(prox_ops_weight_decay[i]);
			prox_ops[i] = NULL;
		}
	}
}


void iter6_adadelta(	const iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_adadelta_conf, _conf);
	assert(NULL != conf);

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}


void iter6_adam(const iter6_conf* _conf,
		const struct nlop_s* nlop,
		long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
		long NO, enum OUT_TYPE out_type[NO],
		int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_adam_conf, _conf);
	assert(NULL != conf);

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}


void iter6_sgd(		const iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_sgd_conf, _conf);
	assert(NULL != conf);

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}


void iter6_iPALM(	const iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	UNUSED(batchsize);

	auto conf = CAST_DOWN(iter6_iPALM_conf, _conf);

	//Compute sizes
	long isize[NI];
	long osize[NO];

	for (int i = 0; i < NI; i++)
		isize[i] = 2 * md_calc_size(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims);

	for (int o = 0; o < NO; o++)
		osize[o] = 2 * md_calc_size(nlop_generic_codomain(nlop, o)->N, nlop_generic_codomain(nlop, o)->dims);

	//create iter operators
	struct iter_nlop_s nlop_iter = NLOP2ITNLOP(nlop);
	struct iter_op_arr_s adj_op_arr = NLOP2IT_ADJ_ARR(nlop);
	struct iter_nlop_s nlop_batch_gen_iter = NLOP2ITNLOP(nlop_batch_gen);

	const struct operator_p_s* prox_ops_weight_decay[NI];

	for (int i = 0; i < NI; i++) {

		if ((0 != conf->INTERFACE.weight_decay) && (NULL == prox_ops[i]) && (IN_OPTIMIZE == in_type[i])) {

			prox_ops_weight_decay[i] = prox_leastsquares_create(nlop_generic_domain(nlop, i)->N, nlop_generic_domain(nlop, i)->dims, conf->INTERFACE.weight_decay, NULL);
			prox_ops[i] = prox_ops_weight_decay[i];

		} else {

			prox_ops_weight_decay[i] = NULL;
		}
	}

	struct iter_op_p_s prox_iter[NI];

	for (int i = 0; i < NI; i++)
		prox_iter[i] = OPERATOR_P2ITOP(prox_ops[i]);

	//compute parameter arrays
	float alpha[NI];
	float beta[NI];
	bool convex[NI];

	for (int i = 0; i < NI; i++) {

		alpha[i] = (NULL == conf->alpha_arr) ? conf->alpha : conf->alpha_arr[i];
		beta[i] = (NULL == conf->beta_arr) ? conf->beta : conf->beta_arr[i];
		convex[i] = (NULL == conf->convex_arr) ? conf->convex : conf->convex_arr[i];
	}

	//gpu ref (dst[i] can be null if batch_gen)
	float* gpu_ref = NULL;

	for (int i = 0; i < NI; i++)
		if (IN_OPTIMIZE == in_type[i])
			gpu_ref = dst[i];

	assert(NULL != gpu_ref);

	float* x_old[NI];

	for (int i = 0; i < NI; i++) {

		if (IN_OPTIMIZE == in_type[i])
			x_old[i] = md_alloc_sameplace(1, isize + i, FL_SIZE, gpu_ref);
		else
			x_old[i] = NULL;
	}


	float lipshitz_constants[NI];

	for (int i = 0; i < NI; i++)
		lipshitz_constants[i] = 1. / conf->INTERFACE.learning_rate;

	bool free_monitor = (NULL == monitor);

	if (free_monitor)
		monitor = monitor_iter6_create(true, false, 0, NULL);

	if (_conf->monitor_averaged_objective)
		monitor6_average_objective(monitor);

	const struct iter_dump_s* dump = NULL;

	if (   (NULL != conf->INTERFACE.dump_filename)
	    && (0 < conf->INTERFACE.dump_mod))
		dump = iter6_dump_default_create(conf->INTERFACE.dump_filename,conf->INTERFACE.dump_mod, nlop, conf->INTERFACE.dump_flag, NI, in_type);

	iPALM(	NI, isize, in_type, dst, x_old,
		NO, osize, out_type,
		numbatches, 0, conf->INTERFACE.epochs,
		select_vecops(gpu_ref),
		alpha, beta, convex, conf->trivial_stepsize, conf->reduce_momentum,
		lipshitz_constants, conf->Lmin, conf->Lmax, conf->Lshrink, conf->Lincrease,
		nlop_iter, adj_op_arr,
		prox_iter,
		conf->INTERFACE.batchnorm_momentum,
		nlop_batch_gen_iter,
		(struct iter_op_s){ NULL, NULL }, monitor, dump);

	if (NULL != conf->INTERFACE.history_filename)
		monitor_iter6_dump_record(monitor, conf->INTERFACE.history_filename);

	if (free_monitor)
		monitor_iter6_free(monitor);

	for (int i = 0; i < NI; i++)
		if(IN_OPTIMIZE == in_type[i])
			md_free(x_old[i]);

	if (NULL != dump)
		iter_dump_free(dump);

	for (int i = 0; i < NI; i++) {

		if (NULL != prox_ops_weight_decay[i]) {

			operator_p_free(prox_ops_weight_decay[i]);
			prox_ops[i] = NULL;
		}
	}
}

void iter6_by_conf(	const iter6_conf* _conf,
			const struct nlop_s* nlop,
			long NI, enum IN_TYPE in_type[NI], const struct operator_p_s* prox_ops[NI], float* dst[NI],
			long NO, enum OUT_TYPE out_type[NO],
			int batchsize, int numbatches, const struct nlop_s* nlop_batch_gen, struct monitor_iter6_s* monitor)
{
	auto conf = CAST_MAYBE(iter6_iPALM_conf, _conf);

	if (NULL != conf) {

		iter6_iPALM(	_conf,
				nlop,
				NI, in_type, prox_ops, dst,
				NO, out_type,
				batchsize, numbatches, nlop_batch_gen, monitor);
		return;
	}

	iter6_sgd_like(	_conf,
			nlop,
			NI, in_type, prox_ops, dst,
			NO, out_type,
			batchsize, numbatches, nlop_batch_gen, monitor);
}

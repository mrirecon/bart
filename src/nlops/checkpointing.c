/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/graph.h"

#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "nlops/nlop.h"
#include "num/ops.h"
#include "num/ops_graph.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "checkpointing.h"


struct checkpoint_s {

	nlop_data_t super;

	const struct nlop_s* nlop;

	int II;
	int OO;

	long loop_size;

	int* DI;
	const long** idims;
	complex float** inputs;
	long* in_offsets;

	complex float** der_in;
	complex float** adj_out;

	int* DO;
	const long** odims;
	long* out_offsets;

	complex float** der_out;
	complex float** adj_in;

	bool clear_mem;
	bool der_once;

	bool* der_requested;
};

DEF_TYPEID(checkpoint_s);

static void checkpoint_free_der(struct checkpoint_s* d)
{
	for (int i = 0; i < d->II * d->OO; i++) {

		md_free(d->adj_out[i]);
		md_free(d->der_out[i]);

		d->adj_out[i] = NULL;
		d->der_out[i] = NULL;
	}

	for (int i = 0; i < d->OO; i++) {

		md_free(d->adj_in[i]);
		d->adj_in[i] = NULL;
	}

	for (int i = 0; i < d->II; i++) {

		md_free(d->der_in[i]);
		d->der_in[i] = NULL;
	}
}

static void checkpoint_free_input(struct checkpoint_s* d)
{
	for (int i = 0; i < d->II; i++) {

		md_free(d->inputs[i]);
		d->inputs[i] = NULL;
	}
}

static void checkpoint_clear_der(const nlop_data_t* _data)
{
	auto data = CAST_DOWN(checkpoint_s, _data);

	checkpoint_free_der(data);
	checkpoint_free_input(data);

	nlop_clear_derivatives(data->nlop);
}


static void checkpoint_save_inputs(struct checkpoint_s* data, int II, const complex float* inputs[II], bool save_inputs)
{
	assert(II == data->II);
	assert(0 < II);
	for (int i = 0; i < data->II; i++) {

		if (save_inputs && (NULL == data->inputs[i]))
			data->inputs[i] = md_alloc_sameplace(data->DI[i], data->idims[i], CFL_SIZE, inputs[0]);
		if (!save_inputs) {

			md_free(data->inputs[i]);
			data->inputs[i] = NULL;

		} else {

			md_copy(data->DI[i], data->idims[i], data->inputs[i], inputs[i], CFL_SIZE);
		}
	}
}

static void checkpoint_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(checkpoint_s, _data);
	assert(data->II + data->OO == N);

	int II = data->II;
	int OO = data->OO;

	bool der = false;
	for (int i = 0; i < data->II; i++)
		for (int o = 0; o < data->OO; o++) {

			data->der_requested[o + OO * i] = nlop_der_requested(_data, i, o);
			der = der || nlop_der_requested(_data, i, o);
		}

	checkpoint_save_inputs(data, data->II, (const complex float**)(args + data->OO), der && data->clear_mem);

	nlop_unset_derivatives(data->nlop);

	bool (*der_requested)[II][OO] = (void*)data->der_requested;

	if ((!data->clear_mem) && (1 == data->loop_size))
		nlop_set_derivatives(data->nlop, II, OO, (*der_requested));

	for (long i = 0; i < data->loop_size; i++) {

		complex float* targs[N];

		for (int j = 0; j < data->OO; j++)
			targs[j] = args[j] + i * data->out_offsets[j];

		for (int j = 0; j < data->II; j++)
			targs[j + data->OO] = args[j + data->OO] + i * data->in_offsets[j];

		nlop_generic_apply_unchecked(data->nlop, N, (void*)targs);
	}

	checkpoint_free_der(data);
}

static void checkpoint_re_evaluate(struct checkpoint_s* d, long idx) 
{
	bool (*der_requested)[d->II][d->OO] = (void*)d->der_requested;
	nlop_set_derivatives(d->nlop, d->II, d->OO, (*der_requested));

	void* args[d->OO + d->II];

	for (int j = 0; j < d->OO; j++)
		args[j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, d->inputs[0]);
	
	for (int j = 0; j < d->II; j++)
		args[d->OO + j] = d->inputs[j] + idx * d->in_offsets[j];

	nlop_generic_apply_unchecked(d->nlop, d->OO + d->II, args);

	for (int j = 0; j < d->OO; j++)
		md_free(args[j]);
}

static void checkpoint_eval_der(struct checkpoint_s* d, int i, const complex float* src)
{
	if (NULL == d->der_in[i])
		d->der_in[i] = md_alloc_sameplace(d->DI[i], d->idims[i], CFL_SIZE, src);
	md_copy(d->DI[i], d->idims[i], d->der_in[i], src, CFL_SIZE);


	int num_ops_par = 0;
	const struct operator_s* der_ops[d->OO];

	complex float* der_out_tmp[d->OO];
	long offset[d->OO];

	for (int j = 0; j < d->OO; j++) {

		if (!d->der_requested[j + d->OO * i])
			continue;

		if (NULL == d->der_out[i + d->II * j])
			d->der_out[i + d->II * j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, src);

		der_ops[num_ops_par] = nlop_get_derivative(d->nlop, j, i)->forward;
		offset[num_ops_par] = d->out_offsets[j];
		der_out_tmp[num_ops_par++] = d->der_out[i + d->II * j];
	}


	for (int j = 0; j < d->loop_size; j++) {

		if (d->clear_mem || (1 < d->loop_size))
			checkpoint_re_evaluate(d, j);

		operator_linops_apply_joined_unchecked(num_ops_par, der_ops, der_out_tmp, src);

		src += d->in_offsets[i];
		for (int k = 0; k < num_ops_par; k++)
			der_out_tmp[k] += offset[k];

		if (d->clear_mem || (1 < d->loop_size))
			nlop_clear_derivatives(d->nlop);
	}
} 


static void checkpoint_der(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);

	if (   (NULL == d->der_in[i])
	    || (NULL == d->der_out[i + d->II * o])
	    || (0. != md_zrmse(d->DI[i], d->idims[i], d->der_in[i], src))) {

		checkpoint_eval_der(d, i, src);
	}

	md_copy(d->DO[o], d->odims[o], dst, d->der_out[i + d->II * o], CFL_SIZE);

	if (d->der_once) {

		md_free(d->der_out[i + d->II * o]);
		d->der_out[i + d->II * o] = NULL;
	}

	for (int o = 0; o < d->OO; o++)
		if (NULL != d->der_out[i + d->II * o])
			return;

	md_free(d->der_in[i]);
	d->der_in[i] = NULL;
}


static void checkpoint_eval_adj(struct checkpoint_s* d, int o, const complex float* src)
{
	if (NULL == d->adj_in[o])
		d->adj_in[o] = md_alloc_sameplace(d->DO[o], d->odims[o], CFL_SIZE, src);

	md_copy(d->DO[o], d->odims[o], d->adj_in[o], src, CFL_SIZE);


	int num_ops_par = 0;
	const struct operator_s* der_ops[d->II];

	complex float* adj_out_tmp[d->II];
	complex float* adj_out_tmp_loop[d->II];
	long offset[d->II];
	unsigned long sum_flag = 0;

	for (int j = 0; j < d->II; j++) {

		if (!d->der_requested[o + d->OO * j])
			continue;

		if (NULL == d->adj_out[j + d->II * o])
			d->adj_out[j + d->II * o] = md_alloc_sameplace(d->DI[j], d->idims[j], CFL_SIZE, src);

		
		der_ops[num_ops_par] = nlop_get_derivative(d->nlop, o, j)->adjoint;
		offset[num_ops_par] = d->in_offsets[j];
		
		if ((0 == d->in_offsets[j]) && (1 < d->loop_size)) {

			adj_out_tmp_loop[num_ops_par] = md_alloc_sameplace(d->DI[j], d->idims[j], CFL_SIZE, src);
			sum_flag |= MD_BIT(num_ops_par);
		} else {

			adj_out_tmp_loop[num_ops_par] = d->adj_out[j + d->II * o];
		}

		adj_out_tmp[num_ops_par++] = d->adj_out[j + d->II * o];
	}


	for (int j = 0; j < d->loop_size; j++) {

		if (d->clear_mem || (1 < d->loop_size))
			checkpoint_re_evaluate(d, j);

		operator_linops_apply_joined_unchecked(num_ops_par, der_ops, (0 < j) ? adj_out_tmp_loop : adj_out_tmp, src);

		src += d->out_offsets[o];
		for (int k = 0; k < num_ops_par; k++) {

			adj_out_tmp_loop[k] += offset[k];

			if ((0 < j) && MD_IS_SET(sum_flag, k)) {

				auto iov = operator_codomain(der_ops[k]);
				md_zadd(iov->N, iov->dims, adj_out_tmp[k], adj_out_tmp[k], adj_out_tmp_loop[k]);
			}
				
		}
			
		if (d->clear_mem || (1 < d->loop_size))
			nlop_clear_derivatives(d->nlop);
	}

	for (int k = 0; k < num_ops_par; k++)
		if (MD_IS_SET(sum_flag, k))
			md_free(adj_out_tmp_loop[k]);
} 


static void checkpoint_adj(const nlop_data_t* _data, int o, int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);

	if (   (NULL == d->adj_in[o])
	    || (NULL == d->adj_out[i + d->II * o])
	    || (0. != md_zrmse(d->DO[o], d->odims[o], src, src))
	    || (0. != md_zrmse(d->DO[o], d->odims[o], d->adj_in[o], d->adj_in[o]))
	    || (0. != md_zrmse(d->DO[o], d->odims[o], d->adj_in[o], src))) {

		checkpoint_eval_adj(d, o, src);
	}

	md_copy(d->DI[i], d->idims[i], dst, d->adj_out[i + d->II * o], CFL_SIZE);

	if (d->der_once) {

		md_free(d->adj_out[i + d->II * o]);
		d->adj_out[i + d->II * o] = NULL;
	}

	for (int i = 0; i < d->II; i++)
		if (NULL != d->adj_out[i + d->II * o])
			return;

	md_free(d->adj_in[o]);
	d->adj_in[o] = NULL;
}




static void checkpoint_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);

	nlop_free(d->nlop);

	checkpoint_free_der(d);
	xfree(d->adj_in);
	xfree(d->adj_out);
	xfree(d->der_in);
	xfree(d->der_out);

	for (int i = 0; i < d->II; i++) {

		md_free(d->inputs[i]);
		xfree(d->idims[i]);
	}
	xfree(d->idims);
	xfree(d->inputs);
	xfree(d->DI);

	for (int i = 0; i < d->OO; i++)
		xfree(d->odims[i]);
	xfree(d->odims);
	xfree(d->DO);

	xfree(d->der_requested);

	xfree(d->in_offsets);
	xfree(d->out_offsets);

	xfree(d);
}

typedef const struct graph_s* (*nlop_graph_t)(const struct operator_s* op, const nlop_data_t* _data);

static const struct graph_s* nlop_graph_checkpointing(const struct operator_s* op, const nlop_data_t* _data)
{
	auto data = CAST_DOWN(checkpoint_s, _data);

	auto subgraph = operator_get_graph(data->nlop->op);
	auto result = create_graph_container(op, "checkpoint container", subgraph);

	return result;
}

static const struct nlop_s* nlop_checkpoint_loop_create(const struct nlop_s* nlop, bool der_once, bool clear_mem, long loop_size, int II, int iloop_dim[II], int OO, int oloop_dim[OO])
{
	PTR_ALLOC(struct checkpoint_s, d);
	SET_TYPEID(checkpoint_s, d);

	assert(II == nlop_get_nr_in_args(nlop));
	assert(OO == nlop_get_nr_out_args(nlop));

	int max_DI = 0;
	int max_DO = 0;

	PTR_ALLOC(int[OO], DO);
	PTR_ALLOC(const long*[OO], odims);
	PTR_ALLOC(int[II], DI);
	PTR_ALLOC(const long*[II], idims);

	for (int i = 0; i < OO; i++) {

		auto iov = nlop_generic_codomain(nlop, i);
		(*DO)[i] = iov->N;
		max_DO = MAX(max_DO, iov->N);

		PTR_ALLOC(long[iov->N], tdims);
		md_copy_dims(iov->N, *tdims, iov->dims);
		
		if (1 < loop_size) {

			assert(1 == (*tdims)[oloop_dim[i]]);
			(*tdims)[oloop_dim[i]] = loop_size;
		}

		(*odims)[i] = *PTR_PASS(tdims);
	}

	for (int i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(nlop, i);
		(*DI)[i] = iov->N;
		max_DI = MAX(max_DI, iov->N);

		PTR_ALLOC(long[iov->N], tdims);
		md_copy_dims(iov->N, *tdims, iov->dims);

		if ((1 < loop_size) && (0 <= iloop_dim[i])) {

			assert(1 == (*tdims)[iloop_dim[i]]);
			(*tdims)[iloop_dim[i]] = loop_size;
		}

		(*idims)[i] = *PTR_PASS(tdims);
	}

	d->DO = *PTR_PASS(DO);
	d->odims = * PTR_PASS(odims);
	d->DI = *PTR_PASS(DI);
	d->idims = * PTR_PASS(idims);
	d->clear_mem = clear_mem || 1 < loop_size;


	d->nlop = nlop_optimize_graph(nlop);

	d->der_once = der_once;

	d->II = II;
	d->OO = OO;

	d->der_requested = *TYPE_ALLOC(bool[OO * II]);

	PTR_ALLOC(complex float*[II], inputs);
	PTR_ALLOC(complex float*[II], der_in);
	PTR_ALLOC(complex float*[OO], adj_in);
	PTR_ALLOC(complex float*[II * OO], der_out);
	PTR_ALLOC(complex float*[II * OO], adj_out);

	d->inputs = *PTR_PASS(inputs);
	d->der_in = *PTR_PASS(der_in);
	d->der_out = *PTR_PASS(der_out);
	d->adj_in = *PTR_PASS(adj_in);
	d->adj_out = *PTR_PASS(adj_out);

	for (int i = 0; i < II; i++) {

		d->inputs[i] = NULL;
		d->der_in[i] = NULL;
	}

	for (int i = 0; i < OO; i++)
		d->adj_in[i] = NULL;

	for (int i = 0; i < II * OO; i++) {

		d->adj_out[i] = NULL;
		d->der_out[i] = NULL;
	}

	long nl_odims[OO][max_DO];
	long nl_idims[II][max_DI];

	for (int i = 0; i < OO; i++){

		md_singleton_dims(max_DO, nl_odims[i]);
		md_copy_dims(d->DO[i], nl_odims[i], d->odims[i]);
	}

	for (int i = 0; i < II; i++){

		md_singleton_dims(max_DI, nl_idims[i]);
		md_copy_dims(d->DI[i], nl_idims[i], d->idims[i]);
	}

	d->loop_size = 1;
	d->in_offsets = *TYPE_ALLOC(long[II]);
	d->out_offsets = *TYPE_ALLOC(long[OO]);
	md_set_dims(II, d->in_offsets, 0);
	md_set_dims(OO, d->out_offsets, 0);

	if (1 < loop_size) {

		d->loop_size = loop_size;

		long nl_ostrs[OO][max_DO];
		long nl_istrs[II][max_DI];

		const long* nl_ostrs2[OO];
		const long* nl_istrs2[II];

		for (int i = 0; i < II; i++) {

			md_calc_strides(max_DI, nl_istrs[i], nl_idims[i], CFL_SIZE);
			nl_istrs2[i] = nl_istrs[i];

			if (0 <= iloop_dim[i])
				d->in_offsets[i] = nl_istrs[i][iloop_dim[i]] / (long)CFL_SIZE;
		}

		for (int o = 0; o < OO; o++) {

			md_calc_strides(max_DO, nl_ostrs[o], nl_odims[o], CFL_SIZE);
			nl_ostrs2[o] = nl_ostrs[o];
			d->out_offsets[o] = nl_istrs[o][oloop_dim[o]] / (long)CFL_SIZE;
		}

		d->nlop = nlop_copy_wrapper_F(OO, nl_ostrs2, II, nl_istrs2, d->nlop);
	}

	nlop_der_fun_t der_funs[II][OO];
	nlop_der_fun_t adj_funs[II][OO];

	for (int i = 0; i < II; i++) {
		for (int o = 0; o < OO; o++) {

			der_funs[i][o] = checkpoint_der;
			adj_funs[i][o] = checkpoint_adj;
		}
	}

	const struct nlop_s* result = nlop_generic_managed_create(	OO, max_DO, nl_odims, II, max_DI, nl_idims, CAST_UP(PTR_PASS(d)),
									checkpoint_fun, der_funs, adj_funs, NULL, NULL, checkpoint_del, checkpoint_clear_der, nlop_graph_checkpointing);


	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			if (linop_is_null(nlop_get_derivative(nlop, o, i)))
				result = nlop_no_der_F(result, o, i);

	for (int i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(nlop, i);
		result = nlop_reshape_in_F(result, i, iov->N, nl_idims[i]);
	}
	for (int o = 0; o < OO; o++) {

		auto iov = nlop_generic_codomain(nlop, o);
		result = nlop_reshape_out_F(result, o, iov->N, nl_odims[o]);
	}

	return result;
}

/**
 * Create a checkpointing container around a nlop
 *
 * When the forward operator is called and clear_mem is set to true, the nlop will not store information related to the derivatives but the inputs of the nlop are stored.
 * When the (adjoint) derivative of the nlop is called the forward operator will be called again to compute all information needed to compute the derivatives.
 * Afterwards, all (adjoint) derivatives with the demanded input are computed and stored together with the input in the container.
 * When another (adjoint) derivative is called, the input is checked for changes. If it is the same as before, the precomputed output is copied to the new output,
 * else, the forward operator is computed again to compute the (adjoint) derivative with the updated input.
 * Checkpointing can reduce memory consumption drastically and the overhead of recomputing the forward operator is compensated by reduced swapping from gpu to cpu memory.
 *
 * When the clear_mem flag is not set, the forward operator of the inner nlop will store information related to the derivatives and no reduction in memory usage is expected.
 * In this case, the difference compared to the plain operator is that still all (adjoint) derivatives with respect to one input are precomputed.
 * This might reduce the memory overhead in the operator_apply_joined_unchecked function.
 *
 * @param nlop
 * @param der_once
 * @param clear_mem
 *
 * @returns Container holding nlop
 */
const struct nlop_s* nlop_checkpoint_create(const struct nlop_s* nlop, bool der_once, bool clear_mem)
{
	return nlop_checkpoint_loop_create(nlop, der_once, clear_mem, 0, nlop_get_nr_in_args(nlop), NULL, nlop_get_nr_out_args(nlop), NULL);
}

/**
 * Create a checkpointing container around a nlop and free the nlop
 *
 * When the forward operator is called and clear_mem is set to true, the nlop will not store information related to the derivatives but the inputs of the nlop are stored.
 * When the (adjoint) derivative of the nlop is called the forward operator will be called again to compute all information needed to compute the derivatives.
 * Afterwards, all (adjoint) derivatives with the demanded input are computed and stored together with the input in the container.
 * When another (adjoint) derivative is called, the input is checked for changes. If it is the same as before, the precomputed output is copied to the new output,
 * else, the forward operator is computed again to compute the (adjoint) derivative with the updated input.
 * Checkpointing can reduce memory consumption drastically and the overhead of recomputing the forward operator is compensated by reduced swapping from gpu to cpu memory.
 *
 * When the clear_mem flag is not set, the forward operator of the inner nlop will store information related to the derivatives and no reduction in memory usage is expected.
 * In this case, the difference compared to the plain operator is that still all (adjoint) derivatives with respect to one input are precomputed.
 * This might reduce the memory overhead in the operator_apply_joined_unchecked function.
 *
 * @param nlop
 * @param der_once
 * @param clear_mem
 *
 * @returns Container holding nlop
 */
const struct nlop_s* nlop_checkpoint_create_F(const struct nlop_s* nlop, bool der_once, bool clear_mem)
{
	auto result = nlop_checkpoint_create(nlop, der_once, clear_mem);
	nlop_free(nlop);
	return result;
}

bool nlop_is_checkpoint(const struct nlop_s* nlop)
{
	const nlop_data_t* _data = nlop_get_data_nested(nlop);

	if (NULL == _data)
		return false;

	auto data = CAST_MAYBE(checkpoint_s, _data);
	
	return NULL != data;
}

const struct nlop_s* nlop_loop_generic_F(int N, const struct nlop_s* nlop, int II, int iloop_dim[II], int OO, int oloop_dim[OO])
{
	auto result = nlop_checkpoint_loop_create(nlop, true, true, N, II, iloop_dim, OO, oloop_dim);

	nlop_free(nlop);
	
	return result;
}

const struct nlop_s* nlop_loop_F(int N, const struct nlop_s* nlop, unsigned long dup_flag, int loop_dim)
{
	int II = nlop_get_nr_in_args(nlop);
	int OO = nlop_get_nr_out_args(nlop);

	assert(8 * (int)sizeof(dup_flag) > II);

	int ildim[II];
	for(int i = 0; i < II; i++)
		ildim[i] = MD_IS_SET(dup_flag, i) ? -1 : loop_dim;
	
	int oldim[OO];
	for(int i = 0; i < OO; i++)
		oldim[i] = loop_dim;
	
	return nlop_loop_generic_F(N, nlop, II, ildim, OO, oldim);
}









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

	INTERFACE(nlop_data_t);

	const struct nlop_s* nlop;

	int II;
	int OO;

	unsigned int* DI;
	const long** idims;
	complex float** inputs;

	complex float** der_in;
	complex float** adj_out;

	unsigned int* DO;
	const long** odims;

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

		if (!save_inputs){

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

	if (!data->clear_mem)
		nlop_set_derivatives(data->nlop, II, OO, (*der_requested));

	nlop_generic_apply_unchecked(data->nlop, N, (void*)args);

	checkpoint_free_der(data);
}

static void checkpoint_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);

	if ((NULL != d->der_in[i]) && (0 == md_zrmse(d->DI[i], d->idims[i], d->der_in[i], src))) {

		assert(NULL != d->der_out[i + d->II * o]);
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

		return;
	}

	if (NULL == d->der_in[i])
		d->der_in[i] = md_alloc_sameplace(d->DI[i], d->idims[i], CFL_SIZE, src);
	md_copy(d->DI[i], d->idims[i], d->der_in[i], src, CFL_SIZE);

	if (d->clear_mem) {

		bool (*der_requested)[d->II][d->OO] = (void*)d->der_requested;
		nlop_set_derivatives(d->nlop, d->II, d->OO, (*der_requested));

		void* args[d->OO + d->II];
		for (int j = 0; j < d->OO; j++)
			args[j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, src);
		for (int j = 0; j < d->II; j++)
			args[d->OO + j] = d->inputs[j];

		nlop_generic_apply_unchecked(d->nlop, d->OO + d->II, (void**)args);

		for (int j = 0; j < d->OO; j++)
			md_free(args[j]);
	}

	int num_ops_par = 0;
	const struct operator_s* der_ops[d->OO];
	void* der_out_tmp[d->OO];

	for (int j = 0; j < d->OO; j++) {

		if (!d->der_requested[j + d->OO * i])
			continue;

		if( NULL == d->der_out[i + d->II * j])
			d->der_out[i + d->II * j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, src);

		der_ops[num_ops_par] = nlop_get_derivative(d->nlop, j, i)->forward;
		der_out_tmp[num_ops_par++] = d->der_out[i + d->II * j];
	}

	operator_linops_apply_joined_unchecked(num_ops_par, der_ops, (complex float**)der_out_tmp, src);
	if (d->clear_mem)
		nlop_clear_derivatives(d->nlop);

	assert(NULL != d->der_out[i + d->II * o]);
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

	return;
}


static void checkpoint_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(checkpoint_s, _data);

	if ((NULL != d->adj_in[o]) && (0 == md_zrmse(d->DO[o], d->odims[o], d->adj_in[o], src))) {

		assert(NULL != d->adj_out[i + d->II * o]);
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

		return;
	}

	if (NULL == d->adj_in[o])
		d->adj_in[o] = md_alloc_sameplace(d->DO[o], d->odims[o], CFL_SIZE, src);
	md_copy(d->DO[o], d->odims[o], d->adj_in[o], src, CFL_SIZE);

	if (d->clear_mem) {

		bool (*der_requested)[d->II][d->OO] = (void*)d->der_requested;
		nlop_set_derivatives(d->nlop, d->II, d->OO, (*der_requested));

		void* args[d->OO + d->II];
		for (int j = 0; j < d->OO; j++)
			args[j] = md_alloc_sameplace(d->DO[j], d->odims[j], CFL_SIZE, src);
		for (int j = 0; j < d->II; j++)
			args[d->OO + j] = d->inputs[j];


		nlop_generic_apply_unchecked(d->nlop, d->OO + d->II, (void**)args);

		for (int j = 0; j < d->OO; j++)
			md_free(args[j]);
	}

	int num_ops_par = 0;
	const struct operator_s* adj_ops[d->II];
	void* adj_out_tmp[d->II];

	for (int j = 0; j < d->II; j++) {

		if (!d->der_requested[o + d->OO * j])
			continue;

		if( NULL == d->adj_out[j + d->II * o])
			d->adj_out[j + d->II * o] = md_alloc_sameplace(d->DI[j], d->idims[j], CFL_SIZE, src);

		adj_ops[num_ops_par] = nlop_get_derivative(d->nlop, o, j)->adjoint;
		adj_out_tmp[num_ops_par++] = d->adj_out[j + d->II * o];
	}

	operator_linops_apply_joined_unchecked(num_ops_par, adj_ops, (complex float**)adj_out_tmp, src);
	if (d->clear_mem)
		nlop_clear_derivatives(d->nlop);

	assert(NULL != d->adj_out[i + d->II * o]);
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
	PTR_ALLOC(struct checkpoint_s, d);
	SET_TYPEID(checkpoint_s, d);

	int II = nlop_get_nr_in_args(nlop);
	int OO = nlop_get_nr_out_args(nlop);

	int max_DI = 0;
	int max_DO = 0;

	PTR_ALLOC(unsigned int[OO], DO);
	PTR_ALLOC(const long*[OO], odims);
	PTR_ALLOC(unsigned int[II], DI);
	PTR_ALLOC(const long*[II], idims);

	for (int i = 0; i < OO; i++) {

		auto iov = nlop_generic_codomain(nlop, i);
		(*DO)[i] = iov->N;
		max_DO = MAX(max_DO, iov->N);

		PTR_ALLOC(long[iov->N], tdims);
		md_copy_dims(iov->N, *tdims, iov->dims);
		(*odims)[i] = *PTR_PASS(tdims);
	}

	for (int i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(nlop, i);
		(*DI)[i] = iov->N;
		max_DI = MAX(max_DI, iov->N);

		PTR_ALLOC(long[iov->N], tdims);
		md_copy_dims(iov->N, *tdims, iov->dims);
		(*idims)[i] = *PTR_PASS(tdims);
	}

	d->DO = *PTR_PASS(DO);
	d->odims = * PTR_PASS(odims);
	d->DI = *PTR_PASS(DI);
	d->idims = * PTR_PASS(idims);
	d->clear_mem = clear_mem;


	d->nlop = nlop_clone(nlop);

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

	nlop_der_fun_t der_funs[II][OO];
	nlop_der_fun_t adj_funs[II][OO];

	for(int i = 0; i < II; i++)
		for(int o = 0; o < OO; o++) {

			der_funs[i][o] = checkpoint_der;
			adj_funs[i][o] = checkpoint_adj;
		}

	const struct nlop_s* result = nlop_generic_managed_create(	OO, max_DO, nl_odims, II, max_DI, nl_idims, CAST_UP(PTR_PASS(d)),
									checkpoint_fun, der_funs, adj_funs, NULL, NULL, checkpoint_del, checkpoint_clear_der, nlop_graph_checkpointing);


	for(int i = 0; i < II; i++)
		for(int o = 0; o < OO; o++)
			if (linop_is_null(nlop_get_derivative(nlop, o, i)))
				result = nlop_no_der_F(result, o, i);

	for (int i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(nlop, i);
		result = nlop_reshape_in_F(result, i, iov->N, iov->dims);
	}
	for (int o = 0; o < OO; o++) {

		auto iov = nlop_generic_codomain(nlop, o);
		result = nlop_reshape_out_F(result, o, iov->N, iov->dims);
	}

	return result;
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


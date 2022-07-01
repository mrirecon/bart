/* Copyright 2018-2021. Uecker Lab. University Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Martin Uecker, Moritz Blumenthal
 */

#include <limits.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "num/multind.h"

#include "num/ops.h"
#include "num/ops_graph.h"
#include "num/iovec.h"
#include "num/flpmath.h"

#include "linops/linop.h"

#include "misc/shrdptr.h"
#include "misc/misc.h"
#include "misc/types.h"
#include "misc/debug.h"
#include "misc/list.h"
#include "misc/graph.h"

#include "nlops/stack.h"
#include "nlops/chain.h"
#include "nlop.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


struct nlop_der_s {

	int OO;
	int II;
	bool* requested;
};

bool nlop_der_requested(const nlop_data_t* data, int i, int o)
{
	int II = data->data_der->II;
	int OO = data->data_der->OO;

	assert(i < II);
	assert(o < OO);

	return (*(bool (*)[II][OO])(data->data_der->requested))[i][o];
}

static void nlop_der_set_requested(const nlop_data_t* data, int i, int o, bool status)
{
	int II = data->data_der->II;
	int OO = data->data_der->OO;

	assert(i < II);
	assert(o < OO);

	(*(bool (*)[II][OO])(data->data_der->requested))[i][o] = status;
}

static void nlop_der_set_all_requested(const nlop_data_t* data, bool status)
{
	int II = data->data_der->II;
	int OO = data->data_der->OO;

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			nlop_der_set_requested(data, i, o, status);

}

static struct nlop_der_s* nlop_data_der_create(int II, int OO) {

	struct nlop_der_s* result = TYPE_ALLOC(struct nlop_der_s);

	result->OO = OO;
	result->II = II;

	if (0 < OO * II) {

		bool (*der_requested)[II][OO] = TYPE_ALLOC(bool[II][OO]);
		result->requested = &(*der_requested)[0][0];

		for (int i = 0; i < II; i++)
			for (int o = 0; o < OO; o++)
				(*der_requested)[i][o] = true;
	}

	return result;
}

static void nlop_der_free(const struct nlop_der_s* der_data)
{
	if (0 < der_data->OO * der_data->II)
		xfree(der_data->requested);

	xfree(der_data);
}



struct nlop_op_data_s {

	INTERFACE(operator_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	nlop_fun_t forward1;
	nlop_gen_fun_t forward;

	nlop_graph_t get_graph;
};

static DEF_TYPEID(nlop_op_data_s);


struct nlop_linop_data_s {

	INTERFACE(linop_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	unsigned int o;
	unsigned int i;

	nlop_der_fun_t deriv;
	nlop_der_fun_t adjoint;
	nlop_der_fun_t normal;
	nlop_p_fun_t norm_inv;
};

static DEF_TYPEID(nlop_linop_data_s);


static void sptr_op_del(const struct shared_ptr_s* sptr)
{
	auto data = CONTAINER_OF(sptr, struct nlop_op_data_s, sptr);

	nlop_der_free(data->data->data_der);

	data->del(data->data);
}

static void sptr_linop_del(const struct shared_ptr_s* sptr)
{
	auto data = CONTAINER_OF(sptr, struct nlop_linop_data_s, sptr);

	nlop_der_free(data->data->data_der);

	data->del(data->data);
}

static void op_fun(const operator_data_t* _data, unsigned int N, void* args[__VLA(N)])
{
	auto data = CAST_DOWN(nlop_op_data_s, _data);

	if (NULL != data->forward1) {

		assert(2 == N);
		data->forward1(data->data, args[0], args[1]);

	} else {

		assert(NULL != data->forward);
		data->forward(data->data, N, *(complex float* (*)[N])args);
	}
}

static const struct graph_s* nlop_get_graph_default(const struct operator_s* op, nlop_data_t* data)
{
	return create_graph_operator(op, data->TYPEID->name);
}

static const struct graph_s* operator_nlop_get_graph(const struct operator_s* op)
{
	auto data = CAST_DOWN(nlop_op_data_s, operator_get_data(op));

	if (NULL != data->get_graph)
		return data->get_graph(op, data->data);
	else
		return nlop_get_graph_default(op, data->data);
}

static const struct graph_s* operator_der_get_graph_default(const struct operator_s* op, const linop_data_t* _data, enum LINOP_TYPE lop_type)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	const char* name = ptr_printf("der (%d, %d)\\n%s\\n%s", data->o, data->i, data->data->TYPEID->name, lop_type_str[lop_type]);

	auto result = create_graph_operator(op, name);

	xfree(name);

	return result;
}

static void op_del(const operator_data_t* _data)
{
	auto data = CAST_DOWN(nlop_op_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	xfree(data);
}

static void lop_der(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->deriv(data->data, data->o, data->i, dst, src);
}

static void lop_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->adjoint(data->data, data->o, data->i, dst, src);
}

static void lop_nrm_inv(const linop_data_t* _data, float lambda, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->norm_inv(data->data, data->o, data->i, lambda, dst, src);
}

static void lop_nrm(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	data->normal(data->data, data->o, data->i, dst, src);
}


static void lop_del(const linop_data_t* _data)
{
	auto data = CAST_DOWN(nlop_linop_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	xfree(data);
}

static void der_not_implemented(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(dst);
	UNUSED(src);

	error("Derivative o=%d, i=%d of %s is not implemented!\n", o, i, _data->TYPEID->name);
}

static void adj_not_implemented(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);
	UNUSED(dst);
	UNUSED(src);

	error("Adjoint derivative o=%d, i=%d of %s is not implemented!\n", o, i, _data->TYPEID->name);
}


struct nlop_s* nlop_generic_managed_create2(	int OO, int ON, const long odims[OO][ON], const long ostr[OO][ON], int II, int IN, const long idims[II][IN], const long istr[II][IN],
						nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO],
						nlop_del_fun_t del,
						nlop_clear_der_fun_t clear_der, nlop_graph_t get_graph)
{
	PTR_ALLOC(struct nlop_s, n);

	PTR_ALLOC(struct nlop_op_data_s, d);
	SET_TYPEID(nlop_op_data_s, d);

	d->data = data;
	d->forward1 = NULL;
	d->forward = forward;
	d->get_graph = get_graph;
	d->del = del;

	d->data->data_der = nlop_data_der_create(II, OO);
	d->data->clear_der = clear_der;

	shared_ptr_init(&d->sptr, sptr_op_del);


//	n->op = operator_create2(ON, odims, ostrs, IN, idims, istrs, CAST_UP(PTR_PASS(d)), op_fun, op_del);

	unsigned int D[OO + II];
	for (int i = 0; i < OO + II; i++)
		D[i] = (i < OO) ? ON : IN;

	const long* dims[OO + II];

	for (int i = 0; i < OO + II; i++)
		dims[i] = (i < OO) ? odims[i] : idims[i - OO];

	const long* strs[OO + II];

	for (int i = 0; i < OO + II; i++)
		strs[i] = (i < OO) ? ostr[i] : istr[i - OO];


	const struct linop_s* (*der)[II?:1][OO?:1] = TYPE_ALLOC(const struct linop_s*[II?:1][OO?:1]);

	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++) {
		for (int o = 0; o < OO; o++) {

			PTR_ALLOC(struct nlop_linop_data_s, d2);
			SET_TYPEID(nlop_linop_data_s, d2);

			d2->data = data;
			d2->del = del;
			d2->deriv = (NULL != deriv) ? ((NULL != deriv[i][o]) ? deriv[i][o] : der_not_implemented) : der_not_implemented;
			d2->adjoint = (NULL != adjoint) ? ((NULL != adjoint[i][o]) ? adjoint[i][o] : adj_not_implemented) : adj_not_implemented;
			d2->normal = (NULL != normal) ? normal[i][o] : NULL;
			d2->norm_inv = (NULL != norm_inv) ? norm_inv[i][o] : NULL;

			d2->o = o;
			d2->i = i;

			shared_ptr_copy(&d2->sptr, &d->sptr);
			d2->sptr.del = sptr_linop_del;

			(*der)[i][o] = linop_with_graph_create2(ON, odims[o], ostr[o], IN, idims[i], istr[i],
								CAST_UP(PTR_PASS(d2)), lop_der, lop_adj,  (NULL != normal) ? lop_nrm : NULL, (NULL != norm_inv) ? lop_nrm_inv : NULL, lop_del,
								operator_der_get_graph_default);
		}
	}

	bool io_flags[OO + II];

	for (int i = 0; i < OO + II; i++)
		io_flags[i] = i < OO;

	n->op = operator_generic_create2(OO + II, io_flags, D, dims, strs, CAST_UP(PTR_PASS(d)), op_fun, op_del, operator_nlop_get_graph);

	return PTR_PASS(n);
}

struct nlop_s* nlop_generic_managed_create(int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
	nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del,
	nlop_clear_der_fun_t clear_der, nlop_graph_t get_graph)
{
	long istrs[II][IN];
	for (int i = 0; i < II; i++)
		md_calc_strides(IN, istrs[i], idims[i], CFL_SIZE);
	long ostrs[OO][ON];
	for (int o = 0; o < OO; o++)
		md_calc_strides(ON, ostrs[o], odims[o], CFL_SIZE);

	return nlop_generic_managed_create2(OO, ON, odims, ostrs, II, IN, idims, istrs, data, forward, deriv, adjoint, normal, norm_inv, del, clear_der, get_graph);
}


struct nlop_s* nlop_generic_create2(	int OO, int ON, const long odims[OO][ON], const long ostr[OO][ON], int II, int IN, const long idims[II][IN], const long istr[II][IN],
					nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO],
					nlop_del_fun_t del)
{
	return nlop_generic_managed_create2(OO, ON, odims, ostr, II, IN, idims, istr, data, forward, deriv, adjoint, normal, norm_inv, del, NULL, NULL);
}

struct nlop_s* nlop_generic_create(int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
	nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del)
{
	long istrs[II?:1][IN?:1];

	for (int i = 0; i < II; i++)
		md_calc_strides(IN, istrs[i], idims[i], CFL_SIZE);

	long ostrs[OO?:1][ON?:1];

	for (int o = 0; o < OO; o++)
		md_calc_strides(ON, ostrs[o], odims[o], CFL_SIZE);

	return nlop_generic_create2(OO, ON, odims, ostrs, II, IN, idims, istrs, data, forward, deriv, adjoint, normal, norm_inv, del);
}



struct nlop_s* nlop_create2(unsigned int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)],
				unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
{
	struct nlop_s* op = nlop_generic_create2(1, ON, (const long(*)[])&odims[0], (const long(*)[])&ostrs[0], 1, IN, (const long(*)[])&idims[0], (const long(*)[])&istrs[0], data, NULL,
					(nlop_der_fun_t[1][1]){ { deriv } }, (nlop_der_fun_t[1][1]){ { adjoint } }, (NULL != normal) ? (nlop_der_fun_t[1][1]){ { normal } } : NULL, (NULL != norm_inv) ? (nlop_p_fun_t[1][1]){ { norm_inv } } : NULL, del);

	auto data2 = CAST_DOWN(nlop_op_data_s, operator_get_data(op->op));

	data2->forward1 = forward;

	return op;
}

struct nlop_s* nlop_create(unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
{
	return nlop_create2(	ON, odims, MD_STRIDES(ON, odims, CFL_SIZE),
				IN, idims, MD_STRIDES(IN, idims, CFL_SIZE),
				data, forward, deriv, adjoint, normal, norm_inv, del);
}


int nlop_get_nr_in_args(const struct nlop_s* op)
{
	return operator_nr_in_args(op->op);
}


int nlop_get_nr_out_args(const struct nlop_s* op)
{
	return operator_nr_out_args(op->op);
}



void nlop_free(const struct nlop_s* op)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	operator_free(op->op);

	const struct linop_s* (*der)[II?:1][OO?:1] = (void*)op->derivative;

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			linop_free((*der)[i][o]);

	xfree(der);
	xfree(op);
}


struct nlop_s* nlop_clone(const struct nlop_s* op)
{
	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	n->op = operator_ref(op->op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*nder)[i][o] = linop_clone((*der)[i][o]);

	n->derivative = &(*PTR_PASS(nder))[0][0];
	return PTR_PASS(n);
}

static const struct operator_s* graph_optimize_operator(const struct operator_s* op)
{
	if (NULL == op)
		return NULL;

	auto graph = operator_get_graph(op);

	int count1;
	int count2 = list_count(graph->nodes);

	do {
		count1 = count2;

		graph = operator_graph_optimize_identity_F(graph);
		graph = operator_graph_optimize_identify_F(graph);
		count2 = list_count(graph->nodes);

	} while(count1 > count2);

	return graph_to_operator_F(graph);
}



const struct nlop_s* nlop_optimize_graph(const struct nlop_s* op)
{
	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	n->op = graph_optimize_operator(op->op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*nder)[i][o] = graph_optimize_linop((*der)[i][o]);

	n->derivative = &(*PTR_PASS(nder))[0][0];

	nlop_free(op);
	
	return PTR_PASS(n);
}


nlop_data_t* nlop_get_data(const struct nlop_s* op)
{
	auto data2 = CAST_MAYBE(nlop_op_data_s, operator_get_data(op->op));

	if (NULL == data2)
		return NULL;
#if 1
	auto data3 = CAST_DOWN(nlop_linop_data_s, linop_get_data(op->derivative[0]));
	assert(data3->data == data2->data);
#endif
	return data2->data;
}


nlop_data_t* nlop_get_data_nested(const struct nlop_s* nlop)
{
	const struct operator_s* op = nlop->op;
	while(NULL != get_in_reshape(op))
		op = get_in_reshape(op);

	auto data2 = CAST_MAYBE(nlop_op_data_s, operator_get_data(op));

	if (NULL == data2)
		return NULL;
#if 1
	// If the derivative is zero, this assertion fails
	if (NULL != linop_get_data_nested(nlop->derivative[0])) {

		auto data3 = CAST_DOWN(nlop_linop_data_s, linop_get_data_nested(nlop->derivative[0]));
		assert(data3->data == data2->data);
	}
#endif
	return data2->data;
}


void nlop_apply(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	operator_apply(op->op, ON, odims, dst, IN, idims, src);
}

void nlop_derivative(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	linop_forward(nlop_get_derivative(op, 0, 0), ON, odims, dst, IN, idims, src);
}

void nlop_adjoint(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	linop_adjoint(nlop_get_derivative(op, 0, 0), ON, odims, dst, IN, idims, src);
}



void nlop_generic_apply_unchecked(const struct nlop_s* op, int N, void* args[N])
{
	operator_generic_apply_unchecked(op->op, N, args);
}

void nlop_generic_apply_select_derivative_unchecked(const struct nlop_s* op, int N, void* args[N], unsigned long out_der_flag, unsigned long in_der_flag)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	assert((unsigned int)II <= CHAR_BIT * sizeof(out_der_flag));
	assert((unsigned int)OO <= CHAR_BIT * sizeof(in_der_flag));

	bool select_der[II?:1][OO?:1];
	bool select_all[II?:1][OO?:1];

	for(int o = 0; o < OO; o++) {
		for(int i = 0; i < II; i++) {

			select_der[i][o] = MD_IS_SET(out_der_flag, o) && MD_IS_SET(in_der_flag, i);
			select_all[i][o] = true;
		}
	}

	nlop_clear_derivatives(op);
	nlop_unset_derivatives(op);
	nlop_set_derivatives(op, II, OO, select_der);

	nlop_generic_apply_unchecked(op, N, args);

	nlop_set_derivatives(op, II, OO, select_all);
}

void nlop_clear_derivatives(const struct nlop_s* nlop)
{
	list_t operators = operator_get_list(nlop->op);

	const struct operator_s* op = list_pop(operators);
	while (NULL != op) {

		auto data = CAST_MAYBE(nlop_op_data_s, operator_get_data(op));

		if (NULL == data) {

			op = list_pop(operators);
			continue;
		}

		if (NULL != data->data->clear_der)
			data->data->clear_der(data->data);

		op = list_pop(operators);
	}

	list_free(operators);
}

void nlop_unset_derivatives(const struct nlop_s* nlop) {

	list_t operators = operator_get_list(nlop->op);

	const struct operator_s* op = list_pop(operators);
	while (NULL != op) {

		auto data = CAST_MAYBE(nlop_op_data_s, operator_get_data(op));

		if (NULL == data) {

			op = list_pop(operators);
			continue;
		}

		nlop_der_set_all_requested(data->data, false);

		op = list_pop(operators);
	}
	list_free(operators);

	nlop_clear_derivatives(nlop);
}

void nlop_set_derivatives(const struct nlop_s* nlop, int II, int OO, bool der_requested[II][OO])
{
	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++) {

			if (!der_requested[i][o])
				continue;

			list_t operators = operator_get_list(nlop_get_derivative(nlop, o, i)->adjoint);

			const struct operator_s* op = list_pop(operators);

			while (NULL != op) {

				auto data = operator_get_linop_data(op);

				if (NULL == data) {

					op = list_pop(operators);
					continue;
				}

				auto linop_der_data = CAST_MAYBE(nlop_linop_data_s, data);

				if (NULL == linop_der_data) {

					op = list_pop(operators);
					continue;
				}

				int op_o = linop_der_data->o;
				int op_i = linop_der_data->i;

				nlop_der_set_requested(linop_der_data->data, op_i, op_o, der_requested[i][o]);

				op = list_pop(operators);
			}

			list_free(operators);
		}
}

const struct linop_s* nlop_get_derivative(const struct nlop_s* op, int o, int i)
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	assert((i < II) && (o < OO));

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	return (*der)[i][o];
}

const struct iovec_s* nlop_generic_domain(const struct nlop_s* op, int i)
{
	return operator_arg_in_domain(op->op, (unsigned int)i);
}

const struct iovec_s* nlop_generic_codomain(const struct nlop_s* op, int o)
{
	return operator_arg_out_codomain(op->op, (unsigned int)o);
}



const struct iovec_s* nlop_domain(const struct nlop_s* op)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	return nlop_generic_domain(op, 0);
}

const struct iovec_s* nlop_codomain(const struct nlop_s* op)
{
	assert(1 == nlop_get_nr_in_args(op));
	assert(1 == nlop_get_nr_out_args(op));

	return nlop_generic_codomain(op, 0);
}



const struct nlop_s* nlop_attach(const struct nlop_s* nop, void* ptr, void (*del)(const void* ptr))
{
	struct nlop_s* nlop = nlop_clone(nop);

	const struct operator_s* op = nlop->op;

	nlop->op = operator_attach(op, ptr, del);

	operator_free(op);

	return nlop;
}


struct flatten_s {

	INTERFACE(nlop_data_t);

	size_t* off;
	const struct nlop_s* op;
};

DEF_TYPEID(flatten_s);

static void flatten_fun(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(flatten_s, _data);

	int OO = nlop_get_nr_out_args(data->op);
	int II = nlop_get_nr_in_args(data->op);

	void* args[OO + II];

	for (int o = 0; o < OO; o++)
		args[o] = (void*)dst + data->off[o];

	for (int i = 0; i < II; i++)
		args[OO + i] = (void*)src + data->off[OO + i];


	nlop_generic_apply_unchecked(data->op, OO + II, args);
}

static void flatten_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	auto data = CAST_DOWN(flatten_s, _data);

	int OO = nlop_get_nr_out_args(data->op);
	int II = nlop_get_nr_in_args(data->op);


	for (int o = 0; o < OO; o++) {

		auto iov = linop_codomain(nlop_get_derivative(data->op, o, 0));

		complex float* tmp = md_alloc_sameplace(iov->N, iov->dims, iov->size, src);

		md_clear(iov->N, iov->dims, (void*)dst + data->off[o], iov->size);

		for (int i = 0; i < II; i++) {

			const struct linop_s* der = nlop_get_derivative(data->op, o, i);

			auto iov2 = linop_domain(der);

			complex float* tmp = (0 == i) ? (void*)dst + data->off[o] : md_alloc_sameplace(iov->N, iov->dims, iov->size, src);

			linop_forward(der,
				iov->N, iov->dims, tmp,
				iov2->N, iov2->dims,
				(void*)src + data->off[OO + i]);
			
			if (0 != i) {

				md_zadd(iov->N, iov->dims, (void*)dst + data->off[o], (void*)dst + data->off[o], tmp);
				md_free(tmp);
			}
		}

		md_free(tmp);
	}
}

static void flatten_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	auto data = CAST_DOWN(flatten_s, _data);

	int OO = nlop_get_nr_out_args(data->op);
	int II = nlop_get_nr_in_args(data->op);

	for (int i = 0; i < II; i++) {

		auto iov = linop_domain(nlop_get_derivative(data->op, 0, i));

		for (int o = 0; o < OO; o++) {	// FIXME

			const struct linop_s* der = nlop_get_derivative(data->op, o, i);

			complex float* tmp = (0 == o) ? (void*)dst + data->off[OO + i] : md_alloc_sameplace(iov->N, iov->dims, iov->size, src);

			linop_adjoint_unchecked(der,
				tmp,
				(void*)src + data->off[o]);

			if (0 != o) {

				md_zadd(iov->N, iov->dims, (void*)dst + data->off[OO + i], (void*)dst + data->off[OO + i], tmp);
				md_free(tmp);
			}
		}
	}
}

static void flatten_del(const nlop_data_t* _data)
{
	auto data = CAST_DOWN(flatten_s, _data);

	nlop_free(data->op);

	xfree(data->off);

	xfree(data);
}




struct nlop_s* nlop_flatten(const struct nlop_s* op)
{
	auto result = nlop_flatten_stacked(op);
	if (NULL != result)
		return (struct nlop_s*)result;

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	if (1 < II) {

		// this uses optimization to apply linops jointly

		op = nlop_clone(op);

		for (int i = 0; i < II; i++)
			op = nlop_flatten_in_F(op, i);

		for (int i = 1; i < II; i++)
			op = nlop_stack_inputs_F(op, 0, 1, 0);
		
		return nlop_flatten_F(op);
	}

	long odims[1] = { 0 };
	long ostrs[] = { CFL_SIZE };
	size_t olast = 0;

	PTR_ALLOC(size_t[OO + II], offs);

	for (int o = 0; o < OO; o++) {

		auto iov = nlop_generic_codomain(op, o);

		assert(CFL_SIZE == iov->size);
		assert((int)iov->N == md_calc_blockdim(iov->N, iov->dims, iov->strs, iov->size));

		odims[0] += md_calc_size(iov->N, iov->dims);
		(*offs)[o] = olast;
		olast = odims[0] * CFL_SIZE;
	}


	long idims[1] = { 0 };
	long istrs[1] = { CFL_SIZE };
	size_t ilast = 0;

	for (int i = 0; i < II; i++) {

		auto iov = nlop_generic_domain(op, i);

		assert(CFL_SIZE == iov->size);
		assert((int)iov->N == md_calc_blockdim(iov->N, iov->dims, iov->strs, iov->size));

		idims[0] += md_calc_size(iov->N, iov->dims);
		(*offs)[OO + i] = ilast;
		ilast = idims[0] * CFL_SIZE;
	}

	PTR_ALLOC(struct flatten_s, data);
	SET_TYPEID(flatten_s, data);

	data->op = nlop_clone(op);
	data->off = *PTR_PASS(offs);

	return nlop_create2(1, odims, ostrs, 1, idims, istrs, CAST_UP(PTR_PASS(data)), flatten_fun, flatten_der, flatten_adj, NULL, NULL, flatten_del);
}

struct nlop_s* nlop_flatten_F(const struct nlop_s* op)
{
	auto result = nlop_flatten(op);
	nlop_free(op);
	return result;
}


const struct nlop_s* nlop_flatten_get_op(struct nlop_s* op)
{
	auto data = CAST_MAYBE(flatten_s, nlop_get_data(op));

	return (NULL == data) ? NULL : data->op;
}

const struct nlop_s* nlop_reshape_in(const struct nlop_s* op, int i, int NI, const long idims[NI])
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	int oNI = nlop_generic_domain(op, i)->N;
	const long* oidims = nlop_generic_domain(op, i)->dims;

	debug_printf(DP_DEBUG4, "nlop_reshape_in %d:\t", i);
	debug_print_dims(DP_DEBUG4, oNI, oidims);
	debug_printf(DP_DEBUG4, "to:\t\t\t");
	debug_print_dims(DP_DEBUG4, NI, idims);

	PTR_ALLOC(struct nlop_s, n);

	n->op = operator_reshape(op->op, OO + i, NI, idims);

	auto der = TYPE_ALLOC(const struct linop_s*[II?:1][OO?:1]);

	n->derivative = &(*der)[0][0];


	for (int ii = 0; ii < II; ii++)
		for (int io = 0; io < OO; io++)
			(*der)[ii][io] = (ii == i) ? linop_reshape_in(nlop_get_derivative(op, io, ii), NI, idims) : linop_clone(nlop_get_derivative(op, io, ii));

	return PTR_PASS(n);
}



const struct nlop_s* nlop_reshape_out(const struct nlop_s* op, int o, int NO, const long odims[NO])
{
	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	int oNO = nlop_generic_codomain(op, o)->N;
	const long* oodims = nlop_generic_codomain(op, o)->dims;

	debug_printf(DP_DEBUG4, "nlop_reshape_out %d:\t", o);
	debug_print_dims(DP_DEBUG4, oNO, oodims);
	debug_printf(DP_DEBUG4, "to:\t\t\t");
	debug_print_dims(DP_DEBUG4, NO, odims);

	PTR_ALLOC(struct nlop_s, n);

	n->op = operator_reshape(op->op, o, NO, odims);

	auto der = TYPE_ALLOC(const struct linop_s*[II?:1][OO?:1]);
	n->derivative = &(*der)[0][0];

	for (int ii = 0; ii < II; ii++)
		for (int io = 0; io < OO; io++)
			(*der)[ii][io] = (io == o) ? linop_reshape_out(nlop_get_derivative(op, io, ii), NO, odims) : linop_clone(nlop_get_derivative(op, io, ii));

	return PTR_PASS(n);
}


const struct nlop_s* nlop_reshape_in_F(const struct nlop_s* op, int i, int NI, const long idims[NI])
{
	auto result = nlop_reshape_in(op, i, NI,idims);
	nlop_free(op);
	return result;
}

const struct nlop_s* nlop_reshape_out_F(const struct nlop_s* op, int o, int NO, const long odims[NO])
{
	auto result = nlop_reshape_out(op, o, NO,odims);
	nlop_free(op);
	return result;
}

const struct nlop_s* nlop_flatten_in_F(const struct nlop_s* op, int i)
{
	auto dom = nlop_generic_domain(op, i);
	return nlop_reshape_in_F(op, i, 1, MD_DIMS(md_calc_size(dom->N, dom->dims)));
}

const struct nlop_s* nlop_flatten_out_F(const struct nlop_s* op, int o)
{
	auto cod = nlop_generic_codomain(op, o);
	return nlop_reshape_out_F(op, o, 1, MD_DIMS(md_calc_size(cod->N, cod->dims)));
}

const struct nlop_s* nlop_append_singleton_dim_in_F(const struct nlop_s* op, int i)
{
	long N = nlop_generic_domain(op, i)->N;

	long dims[N + 1];
	md_copy_dims(N, dims, nlop_generic_domain(op, i)->dims);
	dims[N] = 1;

	return nlop_reshape_in_F(op, i, N + 1, dims);
}

const struct nlop_s* nlop_append_singleton_dim_out_F(const struct nlop_s* op, int o)
{
	long N = nlop_generic_codomain(op, o)->N;

	long dims[N + 1];
	md_copy_dims(N, dims, nlop_generic_codomain(op, o)->dims);
	dims[N] = 1;

	return nlop_reshape_out_F(op, o, N + 1, dims);
}


const struct nlop_s* nlop_no_der(const struct nlop_s* op, int o, int i)
{
	PTR_ALLOC(struct nlop_s, n);

	int II = nlop_get_nr_in_args(op);
	int OO = nlop_get_nr_out_args(op);

	n->op = operator_ref(op->op);

	const struct linop_s* (*der)[II][OO] = (void*)op->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int ii = 0; ii < II; ii++)
		for (int oo = 0; oo < OO; oo++) {

			auto cod = linop_codomain((*der)[ii][oo]);
			auto dom = linop_domain((*der)[ii][oo]);

			(*nder)[ii][oo] = ((i == ii) && (o == oo)) ? linop_null_create(cod->N, cod->dims, dom->N, dom->dims) : linop_clone((*der)[ii][oo]);
		}


	n->derivative = &(*PTR_PASS(nder))[0][0];
	return PTR_PASS(n);
}

const struct nlop_s* nlop_no_der_F(const struct nlop_s* op, int o, int i)
{
	auto result = nlop_no_der(op, o, i);
	nlop_free(op);
	return result;
}

void nlop_debug(enum debug_levels dl, const struct nlop_s* x)
{
	int II = nlop_get_nr_in_args(x);

	debug_printf(dl, "NLOP\ninputs: %d\n", II);

	for (int i = 0; i < II; i++) {

		auto io = nlop_generic_domain(x, i);
		debug_print_dims(dl, io->N, io->dims);
	}

	int OO = nlop_get_nr_out_args(x);

	debug_printf(dl, "outputs: %d\n", OO);

	for (int o = 0; o < OO; o++) {

		auto io = nlop_generic_codomain(x, o);
		debug_print_dims(dl, io->N, io->dims);
	}
}




void nlop_generic_apply2_sameplace(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], const long* ostrs[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const long* istrs[NO], const complex float* src[NI],
	const void* ref)
{
	int N = NO + NI;
	void* args[N];

	for (int i = 0; i < NO; i++) {

		assert(iovec_check(nlop_generic_codomain(op, i), DO[i], odims[i], MD_STRIDES(DO[i], odims[i], CFL_SIZE)));

		args[i] = md_alloc_sameplace(DO[i], odims[i], CFL_SIZE, (NULL == ref) ? dst[i] : ref);
	}

	for (int i = 0; i < NI; i++) {

		assert(iovec_check(nlop_generic_domain(op, i), DI[i], idims[i], MD_STRIDES(DI[i], idims[i], CFL_SIZE)));

		args[NO + i] = md_alloc_sameplace(DI[i], idims[i], CFL_SIZE, (NULL == ref) ? src[i] : ref);

		md_copy2(DI[i], idims[i], MD_STRIDES(DI[i], idims[i], CFL_SIZE), args[NO + i], istrs[i], src[i], CFL_SIZE);
	}

	operator_generic_apply_unchecked(op->op, N, args);

	for (int i = 0; i < NO; i++)
		md_copy2(DO[i], odims[i], ostrs[i], dst[i], MD_STRIDES(DO[i], odims[i], CFL_SIZE), args[i], CFL_SIZE);

	for (int i = 0; i < N; i++)
		md_free(args[i]);
}

void nlop_generic_apply_sameplace(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI],
	const void* ref)
{
	const long* ostrs[NO];
	const long* istrs[NI];

	for (int i = 0; i < NO; i++) {

		ostrs[i] = *TYPE_ALLOC(long[DO[i]]);

		md_calc_strides(DO[i], (long*)ostrs[i], odims[i], CFL_SIZE);
	}

	for (int i = 0; i < NI; i++) {

		istrs[i] = *TYPE_ALLOC(long[DI[i]]);

		md_calc_strides(DI[i], (long*)istrs[i], idims[i], CFL_SIZE);
	}

	nlop_generic_apply2_sameplace(op, NO, DO, odims, ostrs, dst, NI, DI, idims, istrs, src, ref);

	for (int i = 0; i < NO; i++)
		xfree(ostrs[i]);

	for (int i = 0; i < NI; i++)
		xfree(istrs[i]);
}

void nlop_generic_apply(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI])
{
	nlop_generic_apply_sameplace(op, NO, DO, odims, dst, NI, DI, idims, src, NULL);
}

void nlop_generic_apply2(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], const long* ostrs[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const long* istrs[NO], const complex float* src[NI])
{
	nlop_generic_apply2_sameplace(op, NO, DO, odims, ostrs, dst, NI, DI, idims, istrs, src, NULL);
}



void nlop_generic_apply_loop_sameplace(const struct nlop_s* op, unsigned long loop_flags,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI],
	const void* ref)
{
	int D = 0;

	for (int i = 0; i < NO; i++)
		D = MAX(D, DO[i]);

	for (int i = 0; i < NI; i++)
		D = MAX(D, DI[i]);

	assert(D < (int)sizeof(loop_flags) * CHAR_BIT);

	long loop_dims[D];
	md_singleton_dims(D, loop_dims);

	const long* nodims[NO];
	const long* nidims[NI];
	const long* ostrs[NO];
	const long* istrs[NI];

	for (int i = 0; i < NO; i++) {

		nodims[i] = *TYPE_ALLOC(long[DO[i]]);
		ostrs[i] = *TYPE_ALLOC(long[DO[i]]);

		md_select_dims(DO[i], ~loop_flags, (long*)nodims[i], odims[i]);
		md_calc_strides(DO[i], (long*)ostrs[i], odims[i], CFL_SIZE);

		long tloop_dims[DO[i]];
		md_select_dims(DO[i], loop_flags, tloop_dims, odims[i]);

		assert(md_check_compat(DO[i], ~0ul, loop_dims, tloop_dims));
		md_max_dims(DO[i], ~0ul, loop_dims, loop_dims, tloop_dims);
	}

	for (int i = 0; i < NI; i++) {

		nidims[i] = *TYPE_ALLOC(long[DI[i]]);
		istrs[i] = *TYPE_ALLOC(long[DI[i]]);

		md_select_dims(DI[i], ~loop_flags, (long*)nidims[i], idims[i]);
		md_calc_strides(DI[i], (long*)istrs[i], idims[i], CFL_SIZE);

		long tloop_dims[DI[i]];
		md_select_dims(DI[i], loop_flags, tloop_dims, idims[i]);

		assert(md_check_compat(DI[i], ~0ul, loop_dims, tloop_dims));
		md_max_dims(DI[i], ~0ul, loop_dims, loop_dims, tloop_dims);
	}

	long pos[D];
	md_singleton_strides(D, pos);

	do {
		complex float* ndst[NO];
		const complex float* nsrc[NI];

		for (int i = 0; i < NO; i++)
			ndst[i] = &(MD_ACCESS(DO[i], ostrs[i], pos, dst[i]));

		for (int i = 0; i < NI; i++)
			nsrc[i] = &(MD_ACCESS(DI[i], istrs[i], pos, src[i]));

		nlop_generic_apply2_sameplace(op,
			NO, DO, nodims, ostrs, ndst,
			NI, DI, nidims, istrs, nsrc,
			ref);

	} while (md_next(D, loop_dims, loop_flags, pos));

	for (int i = 0; i < NO; i++) {

		xfree(nodims[i]);
		xfree(ostrs[i]);
	}

	for (int i = 0; i < NI; i++) {

		xfree(nidims[i]);
		xfree(istrs[i]);
	}
}

void nlop_generic_apply_loop(const struct nlop_s* op, unsigned long loop_flags,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI])
{
	nlop_generic_apply_loop_sameplace(op, loop_flags, NO, DO, odims, dst, NI, DI, idims, src, NULL);
}

void nlop_export_graph(const char* filename, const struct nlop_s* op)
{
	operator_export_graph_dot(filename, op->op);
}

const struct nlop_s* nlop_copy_wrapper(int OO, const long* ostrs[OO], int II, const long* istrs[II], const struct nlop_s* nlop)
{
	PTR_ALLOC(struct nlop_s, n);

	assert(nlop_get_nr_in_args(nlop) == II);
	assert(nlop_get_nr_out_args(nlop) == OO);

	const long* strs[II + OO];

	for (int i = 0; i < OO; i++)
		strs[i] = ostrs[i];

	for (int i = 0; i < II; i++)
		strs[OO + i] = istrs[i];

	n->op = operator_copy_wrapper(OO + II, strs, nlop->op);

	const struct linop_s* (*der)[II][OO] = (void*)nlop->derivative;

	PTR_ALLOC(const struct linop_s*[II][OO], nder);

	for (int ii = 0; ii < II; ii++)
		for (int oo = 0; oo < OO; oo++) {

			auto lop = (struct linop_s*)((*der)[ii][oo]);
			
			int DO = linop_codomain(lop)->N;
			int DI = linop_domain(lop)->N;

			(*nder)[ii][oo] = linop_copy_wrapper2(DI, istrs[ii], DO, ostrs[oo], lop);
		}

	n->derivative = &(*PTR_PASS(nder))[0][0];
	return PTR_PASS(n);
}

const struct nlop_s* nlop_copy_wrapper_F(int OO, const long* ostrs[OO], int II, const long* istrs[II], const struct nlop_s* nlop)
{
	auto result = nlop_copy_wrapper(OO, ostrs, II, istrs, nlop);
	nlop_free(nlop);
	return result;
}

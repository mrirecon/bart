/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include "num/multind.h"

#include "num/ops.h"

#include "linops/linop.h"

#include "misc/shrdptr.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "nlop.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


struct nlop_op_data_s {

	INTERFACE(operator_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	nlop_fun_t forward;
};

static DEF_TYPEID(nlop_op_data_s);


struct nlop_linop_data_s {

	INTERFACE(linop_data_t);

	nlop_data_t* data;
	nlop_del_fun_t del;

	struct shared_ptr_s sptr;

	nlop_fun_t deriv;
	nlop_fun_t adjoint;
};

static DEF_TYPEID(nlop_linop_data_s);


static void sptr_op_del(const struct shared_ptr_s* sptr)
{
	struct nlop_op_data_s* data = CONTAINER_OF(sptr, struct nlop_op_data_s, sptr);

	data->del(data->data);
}

static void sptr_linop_del(const struct shared_ptr_s* sptr)
{
	struct nlop_linop_data_s* data = CONTAINER_OF(sptr, struct nlop_linop_data_s, sptr);
	data->del(data->data);
}

static void op_fun(const operator_data_t* _data, unsigned int N, void* args[__VLA(N)])
{
	const struct nlop_op_data_s* data = CAST_DOWN(nlop_op_data_s, _data);

	data->forward(data->data, args[0], args[1]);	// FIXME: check
}

static void op_del(const operator_data_t* _data)
{
	const struct nlop_op_data_s* data = CAST_DOWN(nlop_op_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	xfree(data);
}

static void lop_der(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct nlop_linop_data_s* data = CAST_DOWN(nlop_linop_data_s, _data);

	data->deriv(data->data, dst, src);
}

static void lop_adj(const linop_data_t* _data, complex float* dst, const complex float* src)
{
	const struct nlop_linop_data_s* data = CAST_DOWN(nlop_linop_data_s, _data);

	data->adjoint(data->data, dst, src);
}

static void lop_del(const linop_data_t* _data)
{
	const struct nlop_linop_data_s* data = CAST_DOWN(nlop_linop_data_s, _data);

	shared_ptr_destroy(&data->sptr);
	xfree(data);
}

struct nlop_s* nlop_create2(unsigned int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)],
				unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_fun_t deriv, nlop_fun_t adjoint, nlop_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
{
	PTR_ALLOC(struct nlop_s, n);

	PTR_ALLOC(struct nlop_op_data_s, d);
	SET_TYPEID(nlop_op_data_s, d);

	d->data = data;
	d->forward = forward;
	d->del = del;

	shared_ptr_init(&d->sptr, sptr_op_del);


	PTR_ALLOC(struct nlop_linop_data_s, d2);
	SET_TYPEID(nlop_linop_data_s, d2);

	d2->data = data;
	d2->del = del;
	d2->deriv = deriv;
	d2->adjoint = adjoint;

	assert(NULL == normal);
	assert(NULL == norm_inv);

	shared_ptr_copy(&d2->sptr, &d->sptr);
	d2->sptr.del = sptr_linop_del;

	n->op = operator_create2(ON, odims, ostrs, IN, idims, istrs, CAST_UP(PTR_PASS(d)), op_fun, op_del);

	n->derivative = linop_create2(ON, odims, ostrs, IN, idims, istrs, CAST_UP(PTR_PASS(d2)), lop_der, lop_adj, NULL, NULL, lop_del);

	//linop_create
	return PTR_PASS(n);
}

struct nlop_s* nlop_create(unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], nlop_data_t* data,
				nlop_fun_t forward, nlop_fun_t deriv, nlop_fun_t adjoint, nlop_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t del)
{
	return nlop_create2(	ON, odims, MD_STRIDES(ON, odims, CFL_SIZE),
				IN, idims, MD_STRIDES(IN, idims, CFL_SIZE),
				data, forward, deriv, adjoint, normal, norm_inv, del);
}


void nlop_free(const struct nlop_s* op)
{
	operator_free(op->op);
	linop_free(op->derivative);
	xfree(op);
}


nlop_data_t* nlop_get_data(struct nlop_s* op)
{
	struct nlop_op_data_s* data2 = CAST_DOWN(nlop_op_data_s, operator_get_data(op->op));
	return data2->data;
}


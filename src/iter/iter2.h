/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2016-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __ITER2_H
#define __ITER2_H

#include "misc/cppwrap.h"
#include "misc/types.h"

struct linop_s;
struct operator_s;
struct operator_p_s;

#ifndef ITER_OP_DATA_S
#define ITER_OP_DATA_S
typedef struct iter_op_data_s { TYPEID* TYPEID; } iter_op_data;
#endif

struct iter_op_op {

	INTERFACE(iter_op_data);
	const struct operator_s* op;
};

DEF_TYPEID(iter_op_op);

struct iter_op_p_op {

	INTERFACE(iter_op_data);
	const struct operator_p_s* op;
};

DEF_TYPEID(iter_op_p_op);

extern void operator_iter(iter_op_data* data, float* dst, const float* src);
extern void operator_p_iter(iter_op_data* data, float rho, float* dst, const float* src);


// the temporay copy is needed if used in loops
#define STRUCT_TMP_COPY(x) ({ __typeof(x) __foo = (x); __typeof(__foo)* __foo2 = alloca(sizeof(__foo)); *__foo2 = __foo; __foo2; })
#define OPERATOR2ITOP(op) (struct iter_op_s){ (NULL == op) ? NULL : operator_iter, CAST_UP(STRUCT_TMP_COPY(((struct iter_op_op){ { &TYPEID(iter_op_op) }, op }))) }
#define OPERATOR_P2ITOP(op) (struct iter_op_p_s){ (NULL == op) ? NULL : operator_p_iter, CAST_UP(STRUCT_TMP_COPY(((struct iter_op_p_op){ { &TYPEID(iter_op_p_op) }, op }))) }

#ifndef ITER_CONF_S
#define ITER_CONF_S
typedef struct iter_conf_s { TYPEID* TYPEID; } iter_conf;
#endif

struct iter_monitor_s;

typedef void (italgo_fun2_f)(iter_conf* conf,
		const struct operator_s* normaleq_op,
		unsigned int D,
		const struct operator_p_s* prox_ops[__VLA2(D)],
		const struct linop_s* ops[__VLA2(D)],
		const float* biases[__VLA2(D)],
		const struct operator_p_s* xupdate_op,
		long size, float* image, const float* image_adj,
		struct iter_monitor_s* monitor);

typedef italgo_fun2_f* italgo_fun2_t;

italgo_fun2_f iter2_conjgrad;
italgo_fun2_f iter2_ist;
italgo_fun2_f iter2_fista;
italgo_fun2_f iter2_admm;
italgo_fun2_f iter2_pocs;


// use with iter_call_s from iter.h as _conf
italgo_fun2_f iter2_call_iter;


struct iter2_call_s {

	INTERFACE(iter_conf);

	italgo_fun2_t fun;
	iter_conf* _conf;
};

extern DEF_TYPEID(iter2_call_s);


#include "misc/cppwrap.h"


#endif


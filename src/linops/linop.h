/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#ifndef __LINOP_H
#define __LINOP_H

#include "misc/cppwrap.h"
#include "misc/types.h"
#include "num/ops.h"

extern TYPEID linop_data_s;
typedef struct linop_data_s { TYPEID* TYPEID; } linop_data_t;


typedef void (*lop_fun_t)(const linop_data_t* _data, complex float* dst, const complex float* src);
typedef void (*lop_p_fun_t)(const linop_data_t* _data, float lambda, complex float* dst, const complex float* src);
typedef void (*del_fun_t)(const linop_data_t* _data);

enum LINOP_TYPE { LOP_FORWARD, LOP_ADJOINT, LOP_NORMAL, LOP_NORMAL_INV };
typedef const struct graph_s* (*lop_graph_t)(const struct operator_s*, const linop_data_t*, enum LINOP_TYPE);
extern const char* lop_type_str[];

struct operator_s;
struct operator_p_s;

struct linop_s {

	const struct operator_s* forward;
	const struct operator_s* adjoint;
	const struct operator_s* normal;
	const struct operator_p_s* norm_inv;
};

extern struct linop_s* linop_from_ops(
	const struct operator_s* forward,
	const struct operator_s* adjoint,
	const struct operator_s* normal,
	const struct operator_p_s* norm_inv);


extern struct linop_s* linop_with_graph_create(int ON, const long odims[__VLA(ON)], int IN, const long idims[__VLA(IN)], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t del_fun,
				lop_graph_t get_graph);

extern struct linop_s* linop_with_graph_create2(int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)],
				int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)],
				linop_data_t* data, lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal,
				lop_p_fun_t norm_inv, del_fun_t del,
				lop_graph_t get_graph);

extern struct linop_s* linop_create(int ON, const long odims[__VLA(ON)], int IN, const long idims[__VLA(IN)], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern struct linop_s* linop_create2(int ON, const long odims[__VLA(ON)], const long ostr[__VLA(ON)],
				int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern const linop_data_t* linop_get_data(const struct linop_s* ptr);
extern const linop_data_t* linop_get_data_nested(const struct linop_s* ptr);
extern const linop_data_t* operator_get_linop_data(const struct operator_s* op);


extern void linop_free(const struct linop_s* op);


extern void linop_forward(const struct linop_s* op, int DN, const long ddims[__VLA(DN)], complex float* dst,
			int SN, const long sdims[__VLA(SN)], const complex float* src);

extern void linop_adjoint(const struct linop_s* op, int DN, const long ddims[__VLA(DN)], complex float* dst,
			int SN, const long sdims[__VLA(SN)], const complex float* src);

extern void linop_normal(const struct linop_s* op, int N, const long dims[__VLA(N)], complex float* dst, const complex float* src);

extern void linop_pseudo_inv(const struct linop_s* op, float lambda, int DN, const long ddims[__VLA(DN)], complex float* dst,
			int SN, const long sdims[__VLA(SN)], const complex float* src);



extern void linop_forward_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_adjoint_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_normal_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_norm_inv_unchecked(const struct linop_s* op, float lambda, complex float* dst, const complex float* src);

extern struct linop_s* linop_chain(const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_chainN(int N, struct linop_s* x[N]);

extern struct linop_s* linop_chain_FF(const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_chainN_F(int N, struct linop_s* x[N]);

extern struct linop_s* linop_stack(int D, int E, const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_stack_FF(int D, int E, const struct linop_s* a, const struct linop_s* b);

extern struct linop_s* linop_stack_cod(int N, const struct linop_s* lops[N], int stack_dim);
extern struct linop_s* linop_stack_cod_F(int N, const struct linop_s* lops[N], int stack_dim);


struct iovec_s;
extern const struct iovec_s* linop_domain(const struct linop_s* x);
extern const struct iovec_s* linop_codomain(const struct linop_s* x);


extern const struct linop_s* linop_clone(const struct linop_s* x);
extern const struct linop_s* linop_get_adjoint(const struct linop_s* x);
extern const struct linop_s* linop_get_normal(const struct linop_s* x);

extern struct linop_s* linop_loop(int D, const long dims[D], struct linop_s* op);
extern struct linop_s* linop_loop_F(int D, const long dims[D], struct linop_s* op);
extern struct linop_s* linop_copy_wrapper2(int DI, const long istrs[DI], int DO, const long ostrs[DO], const struct linop_s* op);
extern struct linop_s* linop_copy_wrapper(int D, const long istrs[D], const long ostrs[D], const struct linop_s* op);
extern struct linop_s* linop_cpu_wrapper(const struct linop_s* op);
extern struct linop_s* linop_gpu_wrapper(const struct linop_s* op);
extern struct linop_s* linop_vptr_wrapper(struct vptr_hint_s* hint, const struct linop_s* op);
extern struct linop_s* linop_vptr_set_dims_wrapper(const struct linop_s* op, const void* cod_ref, const void* dom_ref, struct vptr_hint_s* hint);

extern struct linop_s* linop_null_create2(int NO, const long odims[NO], const long ostrs[NO], int NI, const long idims[NI], const long istrs[NI]);
extern struct linop_s* linop_null_create(int NO, const long odims[NO], int NI, const long idims[NI]);
extern _Bool linop_is_null(const struct linop_s* lop);

extern struct linop_s* linop_plus(const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_plus_FF(const struct linop_s* a, const struct linop_s* b);

extern struct linop_s* linop_reshape_in(const struct linop_s* op, int NI, const long idims[NI]);
extern struct linop_s* linop_reshape_out(const struct linop_s* op, int NO, const long odims[NO]);

extern struct linop_s* linop_reshape_in_F(const struct linop_s* op, int NI, const long idims[NI]);
extern struct linop_s* linop_reshape_out_F(const struct linop_s* op, int NO, const long odims[NO]);


extern struct linop_s* graph_optimize_linop(const struct linop_s* op);
extern const struct operator_s* graph_optimize_operator_linop_F(const struct operator_s* op);

void operator_linops_apply_joined_unchecked(int N, const struct operator_s* op[N], complex float* dst[N], const complex float* src);

#include "misc/cppwrap.h"

#endif // __LINOP_H


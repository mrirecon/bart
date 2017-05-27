/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#ifndef __LINOP_H
#define __LINOP_H

#include "misc/cppwrap.h"
#include "misc/types.h"

TYPEID linop_data_s;
typedef struct linop_data_s { TYPEID* TYPEID; } linop_data_t;


typedef void (*lop_fun_t)(const linop_data_t* _data, complex float* dst, const complex float* src);
typedef void (*lop_p_fun_t)(const linop_data_t* _data, float lambda, complex float* dst, const complex float* src);
typedef void (*del_fun_t)(const linop_data_t* _data);

struct operator_s;
struct operator_p_s;

struct linop_s {

	const struct operator_s* forward;
	const struct operator_s* adjoint;
	const struct operator_s* normal;
	const struct operator_p_s* norm_inv;
};



extern struct linop_s* linop_create(unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern struct linop_s* linop_create2(unsigned int ON, const long odims[__VLA(ON)], const long ostr[__VLA(ON)],
				unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], linop_data_t* data,
				lop_fun_t forward, lop_fun_t adjoint, lop_fun_t normal, lop_p_fun_t norm_inv, del_fun_t);

extern const linop_data_t* linop_get_data(const struct linop_s* ptr);



extern void linop_free(const struct linop_s* op);


extern void linop_forward(const struct linop_s* op, unsigned int DN, const long ddims[__VLA(DN)], complex float* dst,
			unsigned int SN, const long sdims[__VLA(SN)], const complex float* src);

extern void linop_adjoint(const struct linop_s* op, unsigned int DN, const long ddims[__VLA(DN)], complex float* dst,
			unsigned int SN, const long sdims[__VLA(SN)], const complex float* src);

extern void linop_normal(const struct linop_s* op, unsigned int N, const long dims[__VLA(N)], complex float* dst, const complex float* src);

extern void linop_pseudo_inv(const struct linop_s* op, float lambda, unsigned int DN, const long ddims[__VLA(DN)], complex float* dst, 
			unsigned int SN, const long sdims[__VLA(SN)], const complex float* src);



extern void linop_forward_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_adjoint_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_normal_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_norm_inv_unchecked(const struct linop_s* op, float lambda, complex float* dst, const complex float* src);

extern struct linop_s* linop_chain(const struct linop_s* a, const struct linop_s* b);
extern struct linop_s* linop_chainN(unsigned int N, struct linop_s* x[N]);

struct iovec_s;
extern const struct iovec_s* linop_domain(const struct linop_s* x);
extern const struct iovec_s* linop_codomain(const struct linop_s* x);


extern const struct linop_s* linop_clone(const struct linop_s* x);

extern struct linop_s* linop_loop(unsigned int D, const long dims[D], struct linop_s* op);


// extern const struct linop_s* linop_identity(unsigned int N, const long dims[N]);
// extern const struct linop_s* linop_matrix(unsigned int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const _Complex float* matrix, _Bool use_gpu);
extern const struct linop_s* linop_join(unsigned int D, const struct linop_s* a, const struct linop_s* b);



#include "misc/cppwrap.h"

#endif // __LINOP_H


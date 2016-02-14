/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __OPS_H
#define __OPS_H

#include "misc/cppwrap.h"


typedef void (*operator_fun_t)(const void* _data, unsigned int N, void* args[__VLA(N)]);
typedef void (*operator_p_fun_t)(const void* _data, float mu, _Complex float* _dst, const _Complex float* _src);
typedef void (*operator_del_t)(const void* _data);



struct operator_s;
struct operator_p_s;


// create functions

extern const struct operator_s* operator_create(unsigned int ON, const long out_dims[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)],
		void* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_s* operator_create2(unsigned int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		void* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_p_s* operator_p_create(unsigned int ON, const long out_dims[__VLA(ON)],
			unsigned int IN, const long in_dims[__VLA(IN)], void* data,
			operator_p_fun_t apply, operator_del_t del);

extern const struct operator_p_s* operator_p_create2(unsigned int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		void* data, operator_p_fun_t apply, operator_del_t del);


extern const struct operator_s* operator_generic_create(unsigned int N, const unsigned int D[__VLA(N)], const long* out_dims[__VLA(N)],
		void* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_s* operator_generic_create2(unsigned int N, const unsigned int D[__VLA(N)], const long* out_dims[__VLA(N)], const long* out_strs[__VLA(N)], void* data, operator_fun_t apply, operator_del_t del);



extern const struct operator_s* operator_identity_create(unsigned int N, const long dims[__VLA(N)]);
extern const struct operator_s* operator_identity_create2(unsigned int N, const long dims[__VLA(N)],
					const long ostr[__VLA(N)], const long istr[__VLA(N)]);

extern const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_chainN(unsigned int N, const struct operator_s* ops[__VLA(N)]);

//extern const struct operator_s* operator_mul(const struct operator_s* a, const struct operator_s* b);
//extern const struct operator_s* operator_sum(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_stack(unsigned int D, unsigned int E, const struct operator_s* a, const struct operator_s* b);

// del functions
extern void operator_free(const struct operator_s* x);
extern void operator_p_free(const struct operator_p_s* x);

extern const struct operator_s* operator_ref(const struct operator_s* x);
extern const struct operator_p_s* operator_p_ref(const struct operator_p_s* x);

// apply functions
extern void operator_generic_apply_unchecked(const struct operator_s* op, unsigned int N, void* args[__VLA(N)]);
extern void operator_apply(const struct operator_s* op, unsigned int IN, const long idims[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const _Complex float* src);
extern void operator_apply2(const struct operator_s* op, unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], const _Complex float* src);
extern void operator_p_apply(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const _Complex float* src);
extern void operator_p_apply2(const struct operator_p_s* op, float mu, unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], _Complex float* dst, const long ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], const _Complex float* src);


extern void operator_apply_unchecked(const struct operator_s* op, _Complex float* dst, const _Complex float* src);
extern void operator_p_apply_unchecked(const struct operator_p_s* op, float mu,  _Complex float* dst, const _Complex float* src);


// get functions
struct iovec_s;
extern unsigned int operator_nr_args(const struct operator_s* op);

extern const struct iovec_s* operator_arg_domain(const struct operator_s* op, unsigned int n);
extern const struct iovec_s* operator_domain(const struct operator_s* op);
extern const struct iovec_s* operator_codomain(const struct operator_s* op);

extern const struct iovec_s* operator_p_domain(const struct operator_p_s* op);
extern const struct iovec_s* operator_p_codomain(const struct operator_p_s* op);

extern void* operator_get_data(const struct operator_s* op);
extern void* operator_p_get_data(const struct operator_p_s* x);



#ifdef USE_CUDA
extern const struct operator_s* operator_gpu_wrapper(const struct operator_s* op);
#endif

extern const struct operator_s* operator_loop2(unsigned int N, const unsigned int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op);

#if __GNUC__ < 5
#include "misc/pcaa.h"

#define operator_loop2(N, D, dims, strs, op) \
	operator_loop2(N, D, dims, AR2D_CAST(long, N, D, strs), op)

#endif

extern const struct operator_s* operator_loop(unsigned int D, const long dims[D], const struct operator_s* op);


// iter helper functions
extern void operator_iter(void* _o, float* _dst, const float* _src);
extern void operator_p_iter(void* _o, float lambda, float* _dst, const float* _src);

#include "misc/cppwrap.h"

#endif

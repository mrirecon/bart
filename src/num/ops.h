/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2016-2019. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef __OPS_H
#define __OPS_H

#include "misc/cppwrap.h"
#include "misc/types.h"

typedef struct operator_data_s { TYPEID* TYPEID; } operator_data_t;


typedef void (*operator_fun_t)(const operator_data_t* _data, unsigned int N, void* args[__VLA(N)]);
typedef void (*operator_del_t)(const operator_data_t* _data);



struct operator_s;


// create functions

extern const struct operator_s* operator_create(unsigned int ON, const long out_dims[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)],
		operator_data_t* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_s* operator_create2(unsigned int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		unsigned int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		operator_data_t* data, operator_fun_t apply, operator_del_t del);


extern const struct operator_s* operator_generic_create(unsigned int N, unsigned int io_flags,
		const unsigned int D[__VLA(N)], const long* out_dims[__VLA(N)],
		operator_data_t* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_s* operator_generic_create2(unsigned int N, unsigned int io_flags,
			const unsigned int D[__VLA(N)], const long* out_dims[__VLA(N)], const long* out_strs[__VLA(N)],
			operator_data_t* data, operator_fun_t apply, operator_del_t del);



extern const struct operator_s* operator_identity_create(unsigned int N, const long dims[__VLA(N)]);
extern const struct operator_s* operator_identity_create2(unsigned int N, const long dims[__VLA(N)],
					const long ostr[__VLA(N)], const long istr[__VLA(N)]);


extern const struct operator_s* operator_zero_create(unsigned int N, const long dims[N]);
extern const struct operator_s* operator_zero_create2(unsigned int N, const long dims[N], const long ostrs[N]);
extern const struct operator_s* operator_null_create(unsigned int N, const long dims[N]);
extern const struct operator_s* operator_null_create2(unsigned int N, const long dims[N], const long ostrs[N]);



extern const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_chainN(unsigned int N, const struct operator_s* ops[__VLA(N)]);


//extern const struct operator_s* operator_mul(const struct operator_s* a, const struct operator_s* b);
//extern const struct operator_s* operator_sum(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_stack(unsigned int D, unsigned int E, const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_stack2(int M, const int args[M], const int dims[M], const struct operator_s* a, const struct operator_s* b);

extern const struct operator_s* operator_bind2(const struct operator_s* op, unsigned int arg,
			unsigned int N, const long dims[__VLA(N)], const long strs[__VLA(N)], void* ptr);

extern const struct operator_s* operator_attach(const struct operator_s* op, void* ptr, void (*del)(const void* ptr));

// del functions
extern void operator_free(const struct operator_s* x);

extern const struct operator_s* operator_ref(const struct operator_s* x);
extern const struct operator_s* operator_unref(const struct operator_s* x);

#define OP_PASS(x) (operator_unref(x))


// apply functions
extern void operator_generic_apply_unchecked(const struct operator_s* op, unsigned int N, void* args[__VLA(N)]);
extern void operator_apply(const struct operator_s* op, unsigned int ON, const long odims[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const _Complex float* src);
extern void operator_apply2(const struct operator_s* op, unsigned int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], const _Complex float* src);

extern void operator_apply_unchecked(const struct operator_s* op, _Complex float* dst, const _Complex float* src);


// get functions
struct iovec_s;
extern unsigned int operator_nr_args(const struct operator_s* op);
extern unsigned int operator_nr_in_args(const struct operator_s* op);
extern unsigned int operator_nr_out_args(const struct operator_s* op);
extern unsigned int operator_ioflags(const struct operator_s* op);

extern const struct iovec_s* operator_arg_domain(const struct operator_s* op, unsigned int n);
extern const struct iovec_s* operator_domain(const struct operator_s* op);
extern const struct iovec_s* operator_codomain(const struct operator_s* op);

extern operator_data_t* operator_get_data(const struct operator_s* op);


extern const struct operator_s* operator_copy_wrapper(unsigned int N, const long* strs[N], const struct operator_s* op);


extern const struct operator_s* operator_gpu_wrapper2(const struct operator_s* op, long move_flags);
extern const struct operator_s* operator_gpu_wrapper(const struct operator_s* op);

extern const struct operator_s* operator_loop2(unsigned int N, const unsigned int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op);

const struct operator_s* operator_loop_parallel2(unsigned int N, const unsigned int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op,
				unsigned int flags, _Bool gpu);


extern const struct operator_s* operator_loop(unsigned int D, const long dims[D], const struct operator_s* op);
extern const struct operator_s* operator_loop_parallel(unsigned int D, const long dims[D], const struct operator_s* op, unsigned int parallel, _Bool gpu);


extern const struct operator_s* operator_combi_create(int N, const struct operator_s* x[N]);
extern const struct operator_s* operator_link_create(const struct operator_s* op, unsigned int o, unsigned int i);
extern const struct operator_s* operator_dup_create(const struct operator_s* op, unsigned int a, unsigned int b);
extern const struct operator_s* operator_extract_create(const struct operator_s* op, int a, int N, const long dims[N], const long pos[N]);
extern const struct operator_s* operator_extract_create2(const struct operator_s* op, int a, int Da, const long dimsa[Da], const long strsa[Da], const long pos[Da]);
extern const struct operator_s* operator_permute(const struct operator_s* op, int N, const int perm[N]);


extern _Bool operator_zero_or_null_p(const struct operator_s* op);

#include "misc/cppwrap.h"

#endif

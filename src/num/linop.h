/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>


typedef void (*op_fun_t)(const void* _data, complex float* dst, const complex float* src);
typedef void (*op_p_fun_t)(const void* _data, float lambda, complex float* dst, const complex float* src);
typedef void (*del_fun_t)(const void* _data);

struct operator_s;
struct operator_p_s;

struct linop_s {

	const struct operator_s* forward;
	const struct operator_s* adjoint;
	const struct operator_s* normal;
	const struct operator_p_s* pinverse;
};



extern struct linop_s* linop_create(unsigned int N, const long odims[N], const long idims[N], void* data,
				op_fun_t forward, op_fun_t adjoint, op_fun_t normal, op_p_fun_t pinverse, del_fun_t);

extern struct linop_s* linop_create2(unsigned int N, const long odims[N], const long ostr[N], 
				const long idims[N], const long istrs[N], void* data,
				op_fun_t forward, op_fun_t adjoint, op_fun_t normal, op_p_fun_t pinverse, del_fun_t);

extern const void* linop_get_data(const struct linop_s* ptr);


// FIXME: will go away, use this to
extern struct operator_s* linop2operator_compat(struct linop_s* x);

extern void linop_free(const struct linop_s* op);


extern void linop_forward(const struct linop_s* op, unsigned int N, const long ddims[N], complex float* dst, 
			const long sdims[N], const complex float* src);

extern void linop_adjoint(const struct linop_s* op, unsigned int N, const long ddims[N], complex float* dst, 
			const long sdims[N], const complex float* src);

extern void linop_normal(const struct linop_s* op, unsigned int N, const long ddims[N], complex float* dst, 
			const long sdims[N], const complex float* src);

extern void linop_forward_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_adjoint_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_normal_unchecked(const struct linop_s* op, complex float* dst, const complex float* src);
extern void linop_pinverse_unchecked(const struct linop_s* op, float lambda, complex float* dst, const complex float* src);

extern struct linop_s* linop_chain(const struct linop_s* a, const struct linop_s* b);

struct iovec_s;
extern const struct iovec_s* linop_domain(const struct linop_s* x);
extern const struct iovec_s* linop_codomain(const struct linop_s* x);


extern const struct linop_s* linop_clone(const struct linop_s* x);



// extern const struct linop_s* linop_identity(unsigned int N, const long dims[N]);
// extern const struct linop_s* linop_matrix(unsigned int N, const long out_dims[N], const long in_dims[N], const long matrix_dims[N], const _Complex float* matrix, _Bool use_gpu);
extern const struct linop_s* linop_join(unsigned int D, const struct linop_s* a, const struct linop_s* b);


extern void linop_forward_iter( void* _o, float* _dst, const float* _src );
extern void linop_adjoint_iter( void* _o, float* _dst, const float* _src );
extern void linop_normal_iter( void* _o, float* _dst, const float* _src );
extern void linop_pinverse_iter( void* _o, float lambda, float* _dst, const float* _src );




#ifndef _OPS_P_H
#define _OPS_P_H

#include "misc/cppwrap.h"

struct operator_data_s;
typedef struct operator_data_s operator_data_t;

typedef void (*operator_p_fun_t)(const operator_data_t* _data, float mu, _Complex float* _dst, const _Complex float* _src);
typedef void (*operator_del_t)(const operator_data_t* _data);


struct operator_s;
struct operator_p_s;

extern const struct operator_p_s* operator_p_create(int ON, const long out_dims[__VLA(ON)],
			int IN, const long in_dims[__VLA(IN)], operator_data_t* data,
			operator_p_fun_t apply, operator_del_t del);

extern const struct operator_p_s* operator_p_create2(int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		operator_data_t* data, operator_p_fun_t apply, operator_del_t del);

extern void operator_p_free(const struct operator_p_s* x);
extern const struct operator_p_s* operator_p_ref(const struct operator_p_s* x);


extern const struct operator_p_s* operator_p_pre_chain(const struct operator_s* a, const struct operator_p_s* b);
extern const struct operator_p_s* operator_p_pst_chain(const struct operator_p_s* a, const struct operator_s* b);

extern const struct operator_s* operator_p_bind(const struct operator_p_s* op, float alpha);
extern const struct operator_p_s* operator_p_stack(int A, int B, const struct operator_p_s* a, const struct operator_p_s* b);

extern const struct operator_p_s* operator_p_scale(int N, const long dims[N]);


extern void operator_p_apply(const struct operator_p_s* op, float mu, int ON, const long odims[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const _Complex float* src);
extern void operator_p_apply2(const struct operator_p_s* op, float mu, int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], const _Complex float* src);


extern void operator_p_apply_unchecked(const struct operator_p_s* op, float mu,  _Complex float* dst, const _Complex float* src);

extern const struct operator_p_s* operator_p_reshape_in(const struct operator_p_s* op, int N, const long dims[N]);
extern const struct operator_p_s* operator_p_reshape_out(const struct operator_p_s* op, int N, const long dims[N]);
extern const struct operator_p_s* operator_p_flatten_F(const struct operator_p_s* op);

// get functions
struct iovec_s;

extern const struct iovec_s* operator_p_domain(const struct operator_p_s* op);
extern const struct iovec_s* operator_p_codomain(const struct operator_p_s* op);

extern operator_data_t* operator_p_get_data(const struct operator_p_s* x);

extern const struct operator_s* operator_p_upcast(const struct operator_p_s* op);
extern const struct operator_p_s* operator_p_downcast(const struct operator_s* op);

extern const struct operator_p_s* operator_p_gpu_wrapper(const struct operator_p_s* op);
extern const struct operator_p_s* operator_p_cpu_wrapper(const struct operator_p_s* op);
extern const struct operator_p_s* operator_p_cpu_wrapper_F(const struct operator_p_s* op);

struct vptr_hint_s;
extern const struct operator_p_s* operator_p_vptr_set_dims_wrapper(const struct operator_p_s* op, const void* cod_ref, const void* dom_ref, struct vptr_hint_s* hint);

// functions freeing its arguments
extern const struct operator_p_s* operator_p_pre_chain_FF(const struct operator_s* a, const struct operator_p_s* _b);
extern const struct operator_p_s* operator_p_pst_chain_FF(const struct operator_p_s* _a, const struct operator_s* b);
extern const struct operator_s* operator_p_bind_F(const struct operator_p_s* op, float alpha);
extern const struct operator_p_s* operator_p_stack_FF(int A, int B, const struct operator_p_s* _a, const struct operator_p_s* _b);
extern const struct operator_p_s* operator_p_reshape_in_F(const struct operator_p_s* op, int N, const long dims[N]);
extern const struct operator_p_s* operator_p_reshape_out_F(const struct operator_p_s* op, int N, const long dims[N]);


#include "misc/cppwrap.h"

#endif //  _OPS_P_H


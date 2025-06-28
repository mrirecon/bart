
#ifndef _OPS_H
#define _OPS_H

#include "misc/cppwrap.h"
#include "misc/types.h"

#include <stdint.h>

typedef struct operator_data_s { TYPEID* TYPEID; } operator_data_t;

typedef void (*operator_fun_t)(const operator_data_t* _data, int N, void* args[__VLA(N)]);
typedef void (*operator_del_t)(const operator_data_t* _data);


struct graph_s;
struct operator_s;

typedef const struct graph_s* (*operator_get_graph_t)(const struct operator_s*);

// create functions

extern const struct operator_s* operator_create(int ON, const long out_dims[__VLA(ON)],
		int IN, const long in_dims[__VLA(IN)],
		operator_data_t* data, operator_fun_t apply, operator_del_t del);

extern const struct operator_s* operator_create2(int ON, const long out_dims[__VLA(ON)], const long out_strs[__VLA(ON)],
		int IN, const long in_dims[__VLA(IN)], const long in_strs[__VLA(IN)],
		operator_data_t* data, operator_fun_t apply, operator_del_t del);


extern const struct operator_s* operator_generic_create(int N, const _Bool io_flags[N],
		const int D[__VLA(N)], const long* out_dims[__VLA(N)],
		operator_data_t* data, operator_fun_t apply, operator_del_t del, operator_get_graph_t get_graph);

extern const struct operator_s* operator_generic_create2(int N, const _Bool io_flags[N],
			const int D[__VLA(N)], const long* out_dims[__VLA(N)], const long* out_strs[__VLA(N)],
			operator_data_t* data, operator_fun_t apply, operator_del_t del, operator_get_graph_t get_graph);


extern const struct operator_s* operator_identity_create(int N, const long dims[__VLA(N)]);
extern const struct operator_s* operator_identity_create2(int N, const long dims[__VLA(N)],
					const long ostr[__VLA(N)], const long istr[__VLA(N)]);
extern const struct operator_s* operator_reshape_create(int A, const long out_dims[__VLA(A)], int B, const long in_dims[__VLA(B)]);


extern const struct operator_s* operator_zero_create(int N, const long dims[N]);
extern const struct operator_s* operator_zero_create2(int N, const long dims[N], const long ostrs[N]);
extern const struct operator_s* operator_null_create(int N, const long dims[N]);
extern const struct operator_s* operator_null_create2(int N, const long dims[N], const long ostrs[N]);



extern const struct operator_s* operator_chain(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_chainN(int N, const struct operator_s* ops[__VLA(N)]);
extern const struct operator_s* operator_plus_create(const struct operator_s* a, const struct operator_s* b);


//extern const struct operator_s* operator_mul(const struct operator_s* a, const struct operator_s* b);
//extern const struct operator_s* operator_sum(const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_stack(int D, int E, const struct operator_s* a, const struct operator_s* b);
extern const struct operator_s* operator_stack2(int M, const int args[M], const int dims[M], const struct operator_s* a, const struct operator_s* b);

extern const struct operator_s* operator_bind2(const struct operator_s* op, int arg,
			int N, const long dims[__VLA(N)], const long strs[__VLA(N)], void* ptr);

extern const struct operator_s* operator_attach(const struct operator_s* op, void* ptr, void (*del)(const void* ptr));

// del functions
extern void operator_free(const struct operator_s* x);

extern const struct operator_s* operator_ref(const struct operator_s* x);
extern const struct operator_s* operator_unref(const struct operator_s* x);

#define OP_PASS(x) (operator_unref(x))


// apply functions
extern void operator_generic_apply_unchecked(const struct operator_s* op, int N, void* args[__VLA(N)]);
extern void operator_generic_apply_parallel_unchecked(int D, const struct operator_s* op[__VLA(D)], int N, void* args[__VLA(D)][N], int num_threads);
extern void operator_apply(const struct operator_s* op, int ON, const long odims[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const _Complex float* src);
extern void operator_apply2(const struct operator_s* op, int ON, const long odims[__VLA(ON)], const long ostrs[__VLA(ON)], _Complex float* dst, const long IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], const _Complex float* src);

extern void operator_apply_unchecked(const struct operator_s* op, _Complex float* dst, const _Complex float* src);
extern void operator_apply_parallel_unchecked(int D, const struct operator_s* op[__VLA(D)], _Complex float* dst[__VLA(D)], const _Complex float* src[__VLA(D)], int num_threads);
extern void operator_apply_joined_unchecked(int N, const struct operator_s* op[__VLA(N)], _Complex float* dst[__VLA(N)], const _Complex float* src);


// get functions
struct iovec_s;
extern int operator_nr_args(const struct operator_s* op);
extern int operator_nr_in_args(const struct operator_s* op);
extern int operator_nr_out_args(const struct operator_s* op);

extern const struct iovec_s* operator_arg_domain(const struct operator_s* op, int n);
extern const struct iovec_s* operator_arg_in_domain(const struct operator_s* op, int n);
extern const struct iovec_s* operator_arg_out_codomain(const struct operator_s* op, int n);
extern const struct iovec_s* operator_domain(const struct operator_s* op);
extern const struct iovec_s* operator_codomain(const struct operator_s* op);

enum debug_levels;
void operator_debug(enum debug_levels dl, const struct operator_s* x);

extern operator_data_t* operator_get_data(const struct operator_s* op);
extern const _Bool* operator_get_io_flags(const struct operator_s* op);

extern _Bool check_simple_copy(const struct operator_s* op);

extern const struct operator_s* operator_copy_wrapper(int N, const long* strs[N], const struct operator_s* op);
extern const struct operator_s* operator_copy_wrapper_sameplace(int N, const long* strs[N], const struct operator_s* op, const void* ref);

extern const struct operator_s* operator_gpu_wrapper2(const struct operator_s* op, unsigned long move_flags);
extern const struct operator_s* operator_gpu_wrapper(const struct operator_s* op);
extern const struct operator_s* operator_cpu_wrapper(const struct operator_s* op);

struct vptr_hint_s;
extern const struct operator_s* operator_vptr_wrapper(const struct operator_s* op, struct vptr_hint_s* hint);

extern const struct operator_s* operator_loop2(int N, const int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op);

const struct operator_s* operator_loop_parallel2(int N, const int D,
				const long dims[D], const long (*strs)[D],
				const struct operator_s* op,
				unsigned long flags);


extern const struct operator_s* operator_loop(int D, const long dims[D], const struct operator_s* op);
extern const struct operator_s* operator_loop_parallel(int D, const long dims[D], const struct operator_s* op, unsigned long parallel);


extern const struct operator_s* operator_combi_create(int N, const struct operator_s* x[N]);
extern const struct operator_s* operator_combi_create_FF(int N, const struct operator_s* x[N]);
extern const struct operator_s* operator_link_create(const struct operator_s* op, int o, int i);
extern const struct operator_s* operator_link_create_F(const struct operator_s* op, int o, int i);
extern const struct operator_s* operator_dup_create(const struct operator_s* op, int a, int b);
extern const struct operator_s* operator_dup_create_F(const struct operator_s* op, int a, int b);
extern const struct operator_s* operator_extract_create(const struct operator_s* op, int a, int N, const long dims[N], const long pos[N]);
extern const struct operator_s* operator_extract_create2(const struct operator_s* op, int a, int Da, const long dimsa[Da], const long strsa[Da], const long pos[Da]);
extern const struct operator_s* operator_permute(const struct operator_s* op, int N, const int perm[N]);
extern const struct operator_s* operator_sort_args_F(const struct operator_s* op);
extern const struct operator_s* operator_reshape(const struct operator_s* op, int i, long N, const long dims[__VLA(N)]);
extern const struct operator_s* get_in_reshape(const struct operator_s* op);

extern const struct operator_s* operator_zadd_create(int II, int N, const long dims[__VLA(N)]);
extern _Bool operator_is_zadd(const struct operator_s* op);


extern _Bool operator_identify(const struct operator_s* a, const struct operator_s* b);

extern struct list_s* operator_get_list(const struct operator_s* op);
extern const struct graph_s* operator_get_graph(const struct operator_s* op);

extern const struct operator_s* operator_nograph_wrapper(const struct operator_s* op);
extern const struct operator_s* graph_optimize_operator_F(const struct operator_s* op);

extern _Bool operator_zero_or_null_p(const struct operator_s* op);

#include "misc/cppwrap.h"

#endif

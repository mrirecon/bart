
#include <stdbool.h>

#include "linops/linop.h"

#ifndef NLOP_H
#define NLOP_H

struct nlop_der_s;
struct nlop_data_s;

typedef void (*nlop_clear_der_fun_t)(const struct nlop_data_s* _data);
typedef struct nlop_data_s { TYPEID* TYPEID; nlop_clear_der_fun_t clear_der; const struct nlop_der_s* data_der; } nlop_data_t;

typedef void (*nlop_fun_t)(const nlop_data_t* _data, complex float* dst, const complex float* src);
typedef void (*nlop_p_fun_t)(const nlop_data_t* _data, unsigned int o, unsigned int i, float lambda, complex float* dst, const complex float* src);
typedef void (*nlop_der_fun_t)(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src);

typedef void (*nlop_del_fun_t)(const nlop_data_t* _data);

typedef void (*nlop_gen_fun_t)(const nlop_data_t* _data, int N, complex float* arg[N]);

typedef const struct graph_s* (*nlop_graph_t)(const struct operator_s* op, const nlop_data_t* _data);

struct operator_s;
struct linop_s;

struct nlop_s {

	const struct operator_s* op;
	const struct linop_s** derivative;
};

extern struct nlop_s* nlop_generic_managed_create(	int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
							nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del,
							nlop_clear_der_fun_t clear_der, nlop_graph_t get_graph);
extern struct nlop_s* nlop_generic_managed_create2(	int OO, int NO, const long odims[OO][NO], const long ostr[OO][NO], int II, int IN, const long idims[II][IN], const long istr[II][IN],
							nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del,
							nlop_clear_der_fun_t clear_der, nlop_graph_t get_graph);

extern struct nlop_s* nlop_generic_create(	int OO, int ON, const long odims[OO][ON], int II, int IN, const long idims[II][IN],
						nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del);

extern struct nlop_s* nlop_generic_create2(	int OO, int NO, const long odims[OO][NO], const long ostr[OO][NO], int II, int IN, const long idims[II][IN], const long istr[II][IN],
						nlop_data_t* data, nlop_gen_fun_t forward, nlop_der_fun_t deriv[II][OO], nlop_der_fun_t adjoint[II][OO], nlop_der_fun_t normal[II][OO], nlop_p_fun_t norm_inv[II][OO], nlop_del_fun_t del);




extern struct nlop_s* nlop_create(	unsigned int ON, const long odims[__VLA(ON)], unsigned int IN, const long idims[__VLA(IN)], nlop_data_t* data,
					nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t);

extern struct nlop_s* nlop_create2(	unsigned int ON, const long odims[__VLA(ON)], const long ostr[__VLA(ON)],
					unsigned int IN, const long idims[__VLA(IN)], const long istrs[__VLA(IN)], nlop_data_t* data,
					nlop_fun_t forward, nlop_der_fun_t deriv, nlop_der_fun_t adjoint, nlop_der_fun_t normal, nlop_p_fun_t norm_inv, nlop_del_fun_t);


extern struct nlop_s* nlop_clone(const struct nlop_s* op);
extern void nlop_free(const struct nlop_s* op);

extern nlop_data_t* nlop_get_data(const struct nlop_s* op);
extern nlop_data_t* nlop_get_data_nested(const struct nlop_s* op);
extern _Bool nlop_der_requested(const nlop_data_t* data, int i, int o);
extern void nlop_get_der_array(const nlop_data_t* data, int N, void* arrays[N]);
extern void nlop_data_der_alloc_memory(const nlop_data_t* data, const void* arg);

extern int nlop_get_nr_in_args(const struct nlop_s* op);
extern int nlop_get_nr_out_args(const struct nlop_s* op);


extern void nlop_apply(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src);
extern void nlop_derivative(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src);
extern void nlop_adjoint(const struct nlop_s* op, int ON, const long odims[ON], complex float* dst, int IN, const long idims[IN], const complex float* src);

extern void nlop_generic_apply_unchecked(const struct nlop_s* op, int N, void* args[N]);
extern void nlop_generic_apply_select_derivative_unchecked(const struct nlop_s* op, int N, void* args[N], unsigned long out_der_flag, unsigned long in_der_flag);
extern void nlop_clear_derivatives(const struct nlop_s* nlop);
extern void nlop_unset_derivatives(const struct nlop_s* nlop);
extern void nlop_set_derivatives(const struct nlop_s* nlop, int II, int OO, bool der_requested[II][OO]);

extern const struct linop_s* nlop_get_derivative(const struct nlop_s* op, int o, int i);

extern const struct iovec_s* nlop_generic_domain(const struct nlop_s* op, int i);
extern const struct iovec_s* nlop_generic_codomain(const struct nlop_s* op, int o);


extern const struct nlop_s* nlop_attach(const struct nlop_s* op, void* ptr, void (*del)(const void* ptr));



struct iovec_s;
extern const struct iovec_s* nlop_domain(const struct nlop_s* op);
extern const struct iovec_s* nlop_codomain(const struct nlop_s* op);


extern struct nlop_s* nlop_flatten(const struct nlop_s* op);
extern struct nlop_s* nlop_flatten_F(const struct nlop_s* op);
extern const struct nlop_s* nlop_flatten_get_op(struct nlop_s* op);

enum debug_levels;
extern void nlop_debug(enum debug_levels dl, const struct nlop_s* x);

extern const struct nlop_s* nlop_reshape_out(const struct nlop_s* op, int o, int NO, const long odims[NO]);
extern const struct nlop_s* nlop_reshape_in(const struct nlop_s* op, int i, int NI, const long idims[NI]);
extern const struct nlop_s* nlop_reshape_out_F(const struct nlop_s* op, int o, int NO, const long odims[NO]);
extern const struct nlop_s* nlop_reshape_in_F(const struct nlop_s* op, int i, int NI, const long idims[NI]);

extern const struct nlop_s* nlop_append_singleton_dim_in_F(const struct nlop_s* op, int i);
extern const struct nlop_s* nlop_append_singleton_dim_out_F(const struct nlop_s* op, int o);

extern const struct nlop_s* nlop_flatten_in_F(const struct nlop_s* op, int i);
extern const struct nlop_s* nlop_flatten_out_F(const struct nlop_s* op, int o);

extern const struct nlop_s* nlop_no_der(const struct nlop_s* op, int o, int i);
extern const struct nlop_s* nlop_no_der_F(const struct nlop_s* op, int o, int i);



extern void nlop_generic_apply(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI]);

extern void nlop_generic_apply2(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], const long* ostrs[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const long* istrs[NO], const complex float* src[NI]);

extern void nlop_generic_apply_loop(const struct nlop_s* op, unsigned long loop_flags,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI]);

extern void nlop_generic_apply2_sameplace(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], const long* ostrs[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const long* istrs[NO], const complex float* src[NI],
	const void* ref);

extern void nlop_generic_apply_sameplace(const struct nlop_s* op,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI],
	const void* ref);

extern void nlop_generic_apply_loop_sameplace(const struct nlop_s* op, unsigned long loop_flags,
	int NO, int DO[NO], const long* odims[NO], complex float* dst[NO],
	int NI, int DI[NI], const long* idims[NI], const complex float* src[NI],
	const void* ref);

extern void nlop_export_graph(const char* filename, const struct nlop_s* op);

extern const struct nlop_s* nlop_copy_wrapper(int OO, const long* ostrs[OO], int II, const long* istrs[II], const struct nlop_s* nlop);
extern const struct nlop_s* nlop_copy_wrapper_F(int OO, const long* ostrs[OO], int II, const long* istrs[II], const struct nlop_s* nlop);

extern const struct nlop_s* nlop_optimize_graph(const struct nlop_s* op);

#endif


 
#ifndef _ITER_LSQR_H
#define _ITER_LSQR_H 1

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/itop.h"

#include "misc/cppwrap.h"

struct operator_s;
struct operator_p_s;


struct lsqr_conf {

	float lambda;
	_Bool it_gpu;
	_Bool warmstart;
	itop_continuation_t icont;
};

struct iter_monitor_s;
extern const struct lsqr_conf lsqr_defaults;

extern const struct operator_p_s* lsqr2_create(const struct lsqr_conf* conf,
				      italgo_fun2_t italgo, iter_conf* iconf,
				      const float* init,
				      const struct linop_s* model_op,
				      const struct operator_s* precond_op,
			              int num_funs,
				      const struct operator_p_s* prox_funs[__VLA2(num_funs)],
				      const struct linop_s* prox_linops[__VLA2(num_funs)],
				      struct iter_monitor_s* monitor);

extern const struct operator_p_s* wlsqr2_create(const struct lsqr_conf* conf,
					italgo_fun2_t italgo, iter_conf* iconf,
				        const float* init,
					const struct linop_s* model_op,
					const struct linop_s* weights,
					const struct operator_s* precond_op,
					int num_funs,
					const struct operator_p_s* prox_funs[__VLA2(num_funs)],
					const struct linop_s* prox_linops[__VLA2(num_funs)],
				        struct iter_monitor_s* monitor);



extern void lsqr(	int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[__VLA(N)], _Complex float* x,
			const long y_dims[__VLA(N)], const _Complex float* y,
			const struct operator_s* precond_op);

extern void wlsqr(	int N, const struct lsqr_conf* conf,
			italgo_fun_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			const struct operator_p_s* thresh_op,
			const long x_dims[__VLA(N)], _Complex float* x,
			const long y_dims[__VLA(N)], const _Complex float* y,
			const long w_dims[__VLA(N)], const _Complex float* w,
			const struct operator_s* precond_op);

extern void lsqr2(	int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			int num_funs,
			const struct operator_p_s* prox_funs[__VLA2(num_funs)],
			const struct linop_s* prox_linops[__VLA2(num_funs)],
			const long x_dims[__VLA(N)], _Complex float* x,
			const long y_dims[__VLA(N)], const _Complex float* y,
			const struct operator_s* precond_op,
			struct iter_monitor_s* monitor);

extern void wlsqr2(	int N, const struct lsqr_conf* conf,
			italgo_fun2_t italgo, iter_conf* iconf,
			const struct linop_s* model_op,
			int num_funs,
			const struct operator_p_s* prox_funs[__VLA2(num_funs)],
			const struct linop_s* prox_linops[__VLA2(num_funs)],
			const long x_dims[__VLA(N)], complex float* x,
			const long y_dims[__VLA(N)], const complex float* y,
			const long w_dims[__VLA(N)], const complex float* w,
			const struct operator_s* precond_op);


#include "misc/cppwrap.h"

#endif



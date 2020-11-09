
#include "iter/iter2.h"

#include "misc/nested.h"

struct operator_s;
struct operator_p_s;
struct linop_s;
struct iter_monitor_s;


/**
 * Used to flexibly modify behavior, e.g. continuation
 *
 * @param iconf	config to be modified
 * @param alpha regularization
 */

typedef void CLOSURE_TYPE(itop_continuation_t)(iter_conf* iconf);


const struct operator_s* itop_create(	italgo_fun2_t italgo, iter_conf* iconf,
					_Bool warmstart,
					const float* init,
					const struct operator_s* model_op,
					unsigned int num_funs,
					const struct operator_p_s* prox_funs[num_funs],
					const struct linop_s* prox_linops[num_funs],
					struct iter_monitor_s* monitor,
					itop_continuation_t itop_cont);

const struct operator_p_s* itop_p_create(italgo_fun2_t italgo, iter_conf* iconf,
					_Bool warmstart,
					const float* init,
					const struct operator_s* model_op,
					unsigned int num_funs,
					const struct operator_p_s* prox_funs[num_funs],
					const struct linop_s* prox_linops[num_funs],
					struct iter_monitor_s* monitor,
					itop_continuation_t itop_cont);

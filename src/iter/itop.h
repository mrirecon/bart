
#include "iter/iter2.h"

struct operator_s;
struct operator_p_s;
struct linop_s;
struct iter_monitor_s;

const struct operator_s* itop_create(	italgo_fun2_t italgo, iter_conf* iconf,
					const float* init,
					const struct operator_s* model_op,
					unsigned int num_funs, unsigned int num_pfuns,
					const struct operator_p_s* prox_funs[num_pfuns],
					const struct linop_s* prox_linops[num_funs],
					struct iter_monitor_s* monitor);


#include "misc/types.h"

struct operator_p_s;
struct iter3_conf_s;
struct iter_op_s;
struct iter_nlop_s;
struct nlop_s;

typedef void iter4_fun_f(const struct iter3_conf_s* _conf,
		const struct nlop_s* nlop,
		long N, float* dst, const float* ref,
		long M, const float* src,
		const struct operator_p_s* solve,
		const struct iter_op_s cb);

iter4_fun_f iter4_irgnm;
iter4_fun_f iter4_irgnm2;
iter4_fun_f iter4_landweber;
iter4_fun_f iter4_lbfgs;

struct iter3_irgnm_conf;
extern const struct iter3_irgnm_conf iter3_irgnm_defaults;
extern const struct iter3_landweber_conf iter3_landweber_defaults;
extern const struct iter3_lbfgs_conf iter3_lbfgs_defaults;



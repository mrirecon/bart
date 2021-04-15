
#include "misc/opts.h"

enum ITER6_TRAIN_ALGORITHM {ITER6_NONE, ITER6_SGD, ITER6_ADAM, ITER6_ADADELTA, ITER6_IPALM};
extern enum ITER6_TRAIN_ALGORITHM iter_6_select_algo;

extern struct opt_s iter6_opts[];
extern struct opt_s iter6_sgd_opts[];
extern struct opt_s iter6_adadelta_opts[];
extern struct opt_s iter6_adam_opts[];
extern struct opt_s iter6_ipalm_opts[];

extern const int N_iter6_opts;
extern const int N_iter6_sgd_opts;
extern const int N_iter6_adadelta_opts;
extern const int N_iter6_adam_opts;
extern const int N_iter6_ipalm_opts;

extern void opts_iter6_init(void);
extern struct iter6_conf_s* iter6_get_conf_from_opts(void);
extern void iter6_copy_config_from_opts(struct iter6_conf_s* result);
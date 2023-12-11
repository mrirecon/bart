
#include "linops/someops.h"

struct noir2_conf_s;
struct noir2_s;
struct iter_conjgrad_conf;
struct iter_fista_conf;

extern void model_net_activate_multigpu(void);
extern void model_net_deactivate_multigpu(void);

struct noir2_model_conf_s;

struct noir2_net_s;
struct noir2_net_config_s;


extern struct noir2_net_config_s* noir2_net_config_create(int N,
	const long trj_dims[N],
	const long wgh_dims[N],
	const long bas_dims[N], const _Complex float* basis,
	const long msk_dims[N], const _Complex float* mask,
	const long ksp_dims[N],
	const long cim_dims[N],
	const long img_dims[N],
	const long col_dims[N],
	unsigned long flag,
	struct noir2_model_conf_s* model_conf);

extern void noir2_net_config_free(struct noir2_net_config_s* x);

extern struct noir2_net_s* noir2_net_create(struct noir2_net_config_s* config, int NB);
extern void noir2_net_free(struct noir2_net_s* x);

extern int noir2_net_get_N(struct noir2_net_s* x);
extern void noir2_net_get_img_dims(struct noir2_net_s* x, int N, long img_dims[N]);
extern void noir2_net_get_cim_dims(struct noir2_net_s* x, int N, long cim_dims[N]);


extern const struct nlop_s* noir_decomp_create_s(struct noir2_s* model);
extern const struct nlop_s* noir_join_create_s(struct noir2_s* model);
extern const struct nlop_s* noir_split_create_s(struct noir2_s* model);

extern const struct nlop_s* noir_adjoint_fft_create_s(struct noir2_s* model);
extern const struct nlop_s* noir_adjoint_nufft_create_s(struct noir2_s* model);
extern const struct nlop_s* noir_fft_create_s(struct noir2_s* model);
extern const struct nlop_s* noir_nufft_create_s(struct noir2_s* model);

extern const struct nlop_s* noir_decomp_create(struct noir2_net_s* model);
extern const struct nlop_s* noir_join_create(struct noir2_net_s* model);
extern const struct nlop_s* noir_split_create(struct noir2_net_s* model);

extern const struct nlop_s* noir_cim_create(struct noir2_net_s* model);
extern const struct nlop_s* noir_extract_img_create(struct noir2_net_s* model);
extern const struct nlop_s* noir_set_img_create(struct noir2_net_s* model);
extern const struct nlop_s* noir_set_col_create(struct noir2_net_s* model);

extern const struct nlop_s* noir_gauss_newton_step_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf);
extern const struct nlop_s* noir_sense_recon_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf);

extern const struct nlop_s* noir_adjoint_fft_create(struct noir2_net_s* model);
extern const struct nlop_s* noir_adjoint_nufft_create(struct noir2_net_s* model);

extern const struct nlop_s* noir_fft_create(struct noir2_net_s* model);
extern const struct nlop_s* noir_nufft_create(struct noir2_net_s* model);

extern const struct nlop_s* noir_gauss_newton_iter_create_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf, int iter, float redu, float alpha_min);
extern const struct nlop_s* noir_rtnlinv_iter_create(struct noir2_net_s* model, const struct iter_conjgrad_conf* iter_conf, int iter, int iter_skip, float redu, float alpha_min, float temp_damp);

extern const struct nlop_s* noir_nlinv_regularization_create(struct noir2_net_s* model, unsigned long mask_flags);
extern const struct nlop_s* noir_nlinv_average_coils_create(struct noir2_net_s* model, enum PADDING padding, int window);

extern const struct nlop_s* noir_nlop_dump_create(struct noir2_net_s* model, const char* filename);


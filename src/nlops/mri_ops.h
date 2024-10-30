
void mri_ops_activate_multigpu(void);
void mri_ops_deactivate_multigpu(void);

struct nufft_conf_s;
struct iter_conjgrad_conf;

struct config_nlop_mri_s;
extern struct config_nlop_mri_s* sense_model_config_cart_create(int N, const long ksp_dims[N], const long img_dims[N], const long col_dims[N], const long pat_dims[N]);
extern struct config_nlop_mri_s* sense_model_config_noncart_create(int N,
	const long trj_dims[N], const long wgh_dims[N], const long ksp_dims[N],
	const long cim_dims[N],	const long img_dims[N], const long col_dims[N],
	const long bas_dims[N], const _Complex float* basis,
	struct nufft_conf_s conf);

extern int sense_model_get_N(struct config_nlop_mri_s* model);
extern void sense_model_get_img_dims(struct config_nlop_mri_s* model, int N, long img_dims[N]);
extern void sense_model_get_col_dims(struct config_nlop_mri_s* model, int N, long col_dims[N]);
extern void sense_model_get_cim_dims(struct config_nlop_mri_s* model, int N, long cim_dims[N]);


extern void sense_model_config_free(const struct config_nlop_mri_s* x);


struct sense_model_s;

extern void sense_model_free(const struct sense_model_s* x);

extern struct sense_model_s* sense_model_create(const struct config_nlop_mri_s* config);
extern struct sense_model_s* sense_model_normal_create(const struct config_nlop_mri_s* config);

extern struct sense_model_s* sense_cart_normal_create(int N, const long max_dims[N], const struct config_nlop_mri_s* conf);
extern struct sense_model_s* sense_noncart_normal_create(int N, const long max_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf);

extern const struct nlop_s* nlop_sense_model_set_data_batch_create(int N, const long dims[N], int Nb, struct sense_model_s* models[Nb]);

extern const struct nlop_s* nlop_sense_adjoint_create(int Nb, struct sense_model_s* models[Nb], _Bool output_psf);

extern const struct nlop_s* nlop_sense_normal_create(int Nb, struct sense_model_s* models[Nb]);
extern const struct nlop_s* nlop_sense_normal_inv_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, unsigned long lambda_flags);
extern const struct nlop_s* nlop_sense_dc_prox_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, unsigned long lambda_flags);
extern const struct nlop_s* nlop_sense_dc_grad_create(int Nb, struct sense_model_s* models[Nb], unsigned long lambda_flags);
extern const struct nlop_s* nlop_sense_scale_maxeigen_create(int Nb, struct sense_model_s* models[Nb], int N, const long dims[N]);

extern const struct nlop_s* nlop_mri_normal_create(int Nb, const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_normal_inv_create(int N, const long lam_dims[N], int Nb, const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);
extern const struct nlop_s* nlop_mri_dc_prox_create(int N, const long lam_dims[N], int Nb, const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);

extern const struct nlop_s* nlop_mri_normal_max_eigen_create(int Nb, const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_scale_rss_create(int Nb, const struct config_nlop_mri_s* conf);
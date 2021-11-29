struct config_nlop_mri_s {

	unsigned long coil_flags;
	unsigned long image_flags;
	unsigned long pattern_flags;
	unsigned long batch_flags;
	unsigned long fft_flags;
	unsigned long coil_image_flags;
	unsigned long basis_flags;

	_Bool noncart;
	struct nufft_conf_s* nufft_conf;
};

struct iter_conjgrad_conf;

struct sense_model_s;
extern void sense_model_free(const struct sense_model_s* x);
extern const struct sense_model_s* sense_model_ref(const struct sense_model_s* x);

extern struct sense_model_s* sense_cart_create(int N,
	const long ksp_dims[N], const long img_dims[N], const long col_dims[N], const long pat_dims[N]);

extern struct sense_model_s* sense_noncart_create(int N,
	const long trj_dims[N], const long wgh_dims[N], const long ksp_dims[N],
	const long cim_dims[N],	const long img_dims[N], const long col_dims[N],
	const long bas_dims[N], const _Complex float* basis,
	struct nufft_conf_s conf);

extern struct sense_model_s* sense_cart_normal_create(int N, const long max_dims[N], const struct config_nlop_mri_s* conf);
extern struct sense_model_s* sense_noncart_normal_create(int N, const long max_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf);

extern const struct nlop_s* nlop_sense_model_set_data_batch_create(int N, const long dims[N], int Nb, struct sense_model_s* models[Nb]);

extern int sense_model_get_N(struct sense_model_s* model);
extern void sense_model_get_img_dims(struct sense_model_s* model, int N, long img_dims[N]);
extern void sense_model_get_col_dims(struct sense_model_s* model, int N, long col_dims[N]);
extern void sense_model_get_cim_dims(struct sense_model_s* model, int N, long cim_dims[N]);

extern const struct nlop_s* nlop_sense_adjoint_create(int Nb, struct sense_model_s* models[Nb], _Bool output_psf);

extern const struct nlop_s* nlop_sense_normal_create(int Nb, struct sense_model_s* models[Nb]);
extern const struct nlop_s* nlop_sense_normal_inv_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, unsigned long lambda_flags);
extern const struct nlop_s* nlop_sense_dc_prox_create(int Nb, struct sense_model_s* models[Nb], struct iter_conjgrad_conf* iter_conf, unsigned long lambda_flags);
extern const struct nlop_s* nlop_sense_dc_grad_create(int Nb, struct sense_model_s* models[Nb], unsigned long lambda_flags);
extern const struct nlop_s* nlop_sense_scale_maxeigen_create(int Nb, struct sense_model_s* models[Nb], int N, const long dims[N]);

extern struct config_nlop_mri_s conf_nlop_mri_simple;

extern const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf);

extern const struct nlop_s* nlop_mri_normal_create(int N, const long cim_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_normal_inv_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);
extern const struct nlop_s* nlop_mri_dc_prox_create(int N, const long max_dims[N], const long lam_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf);

extern const struct nlop_s* nlop_mri_normal_max_eigen_create(int N, const long cim_dims[N], int ND, const long psf_dims[ND], const struct config_nlop_mri_s* conf);
extern const struct nlop_s* nlop_mri_scale_rss_create(int N, const long max_dims[N], const struct config_nlop_mri_s* conf);
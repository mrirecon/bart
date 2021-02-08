
struct linop_s;
struct nlop_s;
enum fat_spec;

enum meco_model {
	MECO_WF,
	MECO_WFR2S,
	MECO_WF2R2S,
	MECO_R2S,
	MECO_PHASEDIFF,
	MECO_PI,
};

enum meco_weights_fB0 {
	MECO_IDENTITY,
	MECO_SOBOLEV,
};

extern int set_num_of_coeff(enum meco_model sel_model);
extern long set_PD_flag(enum meco_model sel_model);
extern long set_R2S_flag(enum meco_model sel_model);
extern long set_fB0_flag(enum meco_model sel_model);


extern void meco_calc_fat_modu(int N, const long dims[N], const complex float TE[*], complex float dst[*], enum fat_spec fat_spec);

extern const complex float* meco_get_scaling(struct nlop_s* op);
extern const struct linop_s* meco_get_fB0_trafo(struct nlop_s* op);
extern void meco_forw_fB0(const struct linop_s* op, complex float* dst, const complex float* src);
extern void meco_back_fB0(const struct linop_s* op, complex float* dst, const complex float* src);

extern unsigned int meco_get_weight_fB0_type(struct nlop_s* op);

extern struct nlop_s* nlop_meco_create(int N, const long y_dims[N], const long x_dims[N], const complex float* TE, enum meco_model sel_model, bool real_pd, enum fat_spec fat_spec, float* scale_fB0, _Bool use_gpu);


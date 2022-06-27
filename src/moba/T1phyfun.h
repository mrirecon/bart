
struct linop_s;
struct nlop_s;
struct noir_model_conf_s;


extern const struct linop_s* T1_get_alpha_trafo(struct nlop_s* op);
extern void T1_forw_alpha(const struct linop_s* op, complex float* dst, const complex float* src);
extern void T1_back_alpha(const struct linop_s* op, complex float* dst, const complex float* src);


extern struct nlop_s* nlop_T1_phy_create(int N, const long map_dims[N], const long out_dims[N], const long in_dims[N],
                const long TI_dims[N], const complex float* TI, bool use_gpu);



struct nlop_s;
struct noir_model_conf_s;
struct moba_conf_s;

extern const struct linop_s* bloch_get_alpha_trafo(const struct nlop_s* op);
extern void bloch_forw_alpha(const struct linop_s* op, complex float* dst, const complex float* src);
extern void bloch_back_alpha(const struct linop_s* op, complex float* dst, const complex float* src);

extern struct nlop_s* nlop_bloch_create(int N, const long der_dims[N], const long map_dims[N], const long out_dims[N], const long in_dims[N],
		const complex float* b1, const struct moba_conf_s* _data, bool use_gpu);



struct lsqr_conf;

struct lad_conf {

	int rwiter;
	float gamma;

	unsigned int wflags;
	
	const struct lsqr_conf* lsqr_conf;
};

extern void lad(unsigned int N, const struct lad_conf* conf,
		italgo_fun_t italgo, iter_conf* iconf,
		const struct linop_s* model_op,
		const struct operator_p_s* thresh_op,
		const long x_dims[N], complex float* x, 
		const long y_dims[N], const complex float* y);

extern void lad2(unsigned int N, const struct lad_conf* conf,
		italgo_fun2_t italgo, iter_conf* iconf,
		const struct linop_s* model_op,
		unsigned int num_funs,
		const struct operator_p_s* thresh_op[num_funs],
		const struct linop_s* thresh_funs[num_funs],
		const long x_dims[N], complex float* x,
		const long y_dims[N], const complex float* y);

extern const struct operator_s* lad2_create(const struct lad_conf* conf,
		italgo_fun2_t italgo, iter_conf* iconf,
		const float* init,
		const struct linop_s* model_op,
		unsigned int num_funs,
		const struct operator_p_s* thresh_op[num_funs],
		const struct linop_s* thresh_funs[num_funs]);


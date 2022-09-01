



struct linop_s;
struct nlop_s;
struct noir_model_conf_s;
struct moba_conf_s;

#ifndef MOBA_MOD
#define MOBA_MOD
struct mobamod {

	struct nlop_s* nlop;
	const struct linop_s* linop;
        const struct linop_s* linop_alpha;
};
#endif

extern struct mobamod moba_create(const long dims[DIMS], const complex float* mask, const complex float* T1, const complex float* b1,
		const complex float* psf, const struct noir_model_conf_s* conf, struct moba_conf_s* data, _Bool use_gpu);



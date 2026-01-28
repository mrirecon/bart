
#include <complex.h>

#include "misc/mri.h"
#include "moba/meco.h"


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

enum seq_type {
	IR_LL,
	MPL,
	TSE,
	MGRE,
	DIFF,
	IR,
	SIM
};

struct mobafit_model_config {

	enum seq_type seq;
	enum meco_model mgre_model;
};


extern struct mobamod moba_create(const long dims[DIMS], const complex float* mask, const complex float* T1, const complex float* TE, const complex float* b1,
		const complex float* b0, const float* scale_fB0, const complex float* psf, const struct noir_model_conf_s* conf, struct moba_conf_s* data);

const struct nlop_s* moba_get_nlop(struct mobafit_model_config* data, const long out_dims[DIMS], const long param_dims[DIMS], const long enc_dims[DIMS], complex float* enc);


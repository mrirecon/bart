#include <complex.h>

#include "misc/mri.h"
#include "moba/meco.h"

struct linop_s;
struct nlop_s;

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


const struct nlop_s* moba_get_nlop(struct mobafit_model_config* data, const long out_dims[DIMS], const long param_dims[DIMS], const long enc_dims[DIMS], complex float* enc);
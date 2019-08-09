


#include "misc/mri.h"

struct linop_s;
struct nlop_s;
struct noir_model_conf_s;

struct T1_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
};


extern struct T1_s T1_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf, const struct noir_model_conf_s* conf, _Bool use_gpu);



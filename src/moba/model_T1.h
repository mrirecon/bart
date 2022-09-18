
#include <complex.h>

#include "misc/mri.h"

struct linop_s;
struct nlop_s;
struct noir_model_conf_s;


#ifndef MOBA_MOD
#define MOBA_MOD
struct mobamod {

	struct nlop_s* nlop;
	const struct linop_s* linop;
        const struct linop_s* linop_alpha;
};
#endif


extern struct mobamod T1_create(const long dims[DIMS], const complex float* mask, const complex float* TI, const complex float* psf,
				float scaling_M0, float scaling_R1s, const struct noir_model_conf_s* conf, float fov);




#include "misc/mri.h"

struct linop_s;
struct nlop_s;
struct noir_model_conf_s;
enum meco_model;
enum fat_spec;

struct meco_s {

	struct nlop_s* nlop;
	const struct linop_s* linop;
	const struct linop_s* linop_fB0;
	const complex float* scaling;
	unsigned int weight_fB0_type;
};

extern struct meco_s meco_create(const long dims[DIMS], const long y_dims[DIMS], const long x_dims[DIMS], const complex float* mask, const complex float* TE, const complex float* psf, enum meco_model sel_model, bool real_pd, enum fat_spec fat_spec, const float* scale_fB0, bool use_gpu, const struct noir_model_conf_s* conf);

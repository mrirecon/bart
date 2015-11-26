
#include "misc/mri.h"
#include "num/ops.h"


extern const struct operator_p_s* prox_dfwavelet_create(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, _Bool use_gpu);


extern const struct operator_p_s* prox_4pt_dfwavelet_create(const long im_dims[DIMS], const long min_size[3], const complex float res[3], unsigned int flow_dim, float lambda, _Bool use_gpu);

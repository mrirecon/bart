#ifndef NN_OPS_H
#define NN_OPS_H

#include "nn/layers.h"

extern const struct nlop_s* nlop_rand_mask_create(int N, const long dims[__VLA(N)], float p);
extern const struct nlop_s* nlop_rand_split_create(int N, const long dims[__VLA(N)], unsigned long shared_dims_flag, float p);
extern const struct nlop_s* nlop_maxpool_create(int N, const long dims[__VLA(N)], const long pool_size[__VLA(N)]);
extern const struct nlop_s* nlop_dropout_create(int N, const long dims[__VLA(N)], float p, unsigned int shared_dims_flag);
extern const struct nlop_s* nlop_noise_create(int N, const long dims[__VLA(N)], float sigma, unsigned long shared_dims_flag, unsigned long shared_sigma_flag);
extern const struct nlop_s* nlop_add_noise_create(int N, const long dims[__VLA(N)], float sigma, unsigned long shared_dims_flag, unsigned long shared_sigma_flag);

enum norm { NORM_NONE, NORM_MAX, NORM_L2 };
extern const struct nlop_s* nlop_norm_create(int N, const long dims[__VLA(N)], unsigned long batch_flag, enum norm norm, _Bool stop_grad);
extern const struct nlop_s* nlop_norm_max_abs_create(int N, const long dims[__VLA(N)], unsigned long batch_flag);
extern const struct nlop_s* nlop_norm_znorm_create(int N, const long dims[__VLA(N)], unsigned long batch_flag);

#endif

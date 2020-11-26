#ifndef BATCHNORM_H
#define BATCHNORM_H
#include "nn/layers.h"
enum NETWORK_STATUS;
extern const struct nlop_s* nlop_stats_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern const struct nlop_s* nlop_normalize_create(int N, const long dims[__VLA(N)], unsigned long flags, float epsilon);
extern const struct nlop_s* nlop_batchnorm_create(int N, const long dims[N], unsigned long flags, float epsilon, enum NETWORK_STATUS status);
#endif

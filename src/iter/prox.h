 
#ifndef __PROX_H
#define __PROX_H

#include "misc/cppwrap.h"

struct operator_p_s;
extern const struct operator_p_s* prox_leastsquares_create(int N, const long dims[__VLA(N)], float lambda, const _Complex float* y);
extern const struct operator_p_s* prox_weighted_leastsquares_create(int N, const long dims[__VLA(N)], float lambda, const _Complex float* y, unsigned long flags, const _Complex float* W);
extern const struct operator_p_s* prox_l2norm_create(int N, const long dims[__VLA(N)], float lambda);
extern const struct operator_p_s* prox_l2ball_create(int N, const long dims[__VLA(N)], float eps, const _Complex float* center);
extern const struct operator_p_s* prox_zero_create(int N, const long dims[__VLA(N)]);
extern const struct operator_p_s* prox_lesseq_create(int N, const long dims[__VLA(N)], const _Complex float* b);
extern const struct operator_p_s* prox_greq_create(int N, const long dims[__VLA(N)], const _Complex float* b);
extern const struct operator_p_s* prox_rvc_create(int N, const long dims[__VLA(N)]);
extern const struct operator_p_s* prox_nonneg_create(int N, const long dims[__VLA(N)]);
extern const struct operator_p_s* prox_zsmax_create(int N, const long dims[__VLA(N)], float a);

#include "misc/cppwrap.h"
#endif

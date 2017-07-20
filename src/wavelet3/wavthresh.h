
#include <stdbool.h>

extern const struct operator_p_s* prox_wavelet3_thresh_create(unsigned int N, const long dims[N], unsigned int flags, unsigned int jflags, const long minsize[N], float lambda, bool randshift);

extern const struct operator_p_s* prox_wavelet3_niht_thresh_create(unsigned int N, const long dims[N], unsigned int flags, unsigned int jflags, const long minsize[N], unsigned int k, bool randshift);

extern const struct operator_p_s* prox_wavelet3_niht_support_create(unsigned int N, const long dims[N], unsigned int flags, unsigned int jflags, const long minsize[N], unsigned int k, bool randshift);




#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"


struct operator_s;

extern const struct linop_s* linop_avg_create(const long imgd_dims[DIMS], unsigned long flags);
extern const struct linop_s* linop_sum_create(const long imgd_dims[DIMS], unsigned long flags);

#ifdef __cplusplus
}
#endif


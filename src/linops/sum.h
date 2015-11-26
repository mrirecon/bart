

#ifdef __cplusplus
extern "C" {
#endif

#include "misc/mri.h"


struct operator_s;

extern const struct linop_s* sum_create(const long imgd_dims[DIMS], _Bool use_gpu);

#ifdef __cplusplus
}
#endif


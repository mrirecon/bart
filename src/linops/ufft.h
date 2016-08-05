
#include <complex.h>


#include "misc/mri.h"

struct operator_s;

// Undersampled fft operator
extern const struct linop_s* linop_ufft_create(const long ksp_dims[DIMS], const long pat_dims[DIMS], const complex float* pat, unsigned int flags, _Bool use_gpu);

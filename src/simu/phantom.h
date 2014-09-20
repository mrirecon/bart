
#include "misc/mri.h"

extern void calc_phantom(const long dims[DIMS], complex float* out, _Bool ksp);
extern void calc_phantom_noncart(const long dims[3], complex float* out, const complex float* traj);
extern void calc_sens(const long dims[DIMS], complex float* sens);
extern void calc_circ(const long dims[DIMS], complex float* img, _Bool ksp);
extern void calc_ring(const long dims[DIMS], complex float* img, _Bool ksp);


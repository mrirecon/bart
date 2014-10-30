
#include "misc/mri.h"

extern void scc(const long out_dims[DIMS], complex float* out_data, const long caldims[DIMS], const complex float* cal_data);
extern void gcc(const long out_dims[DIMS], complex float* out_data, const long caldims[DIMS], const complex float* cal_data);
extern void ecc(const long out_dims[DIMS], complex float* out_data, const long caldims[DIMS], const complex float* cal_data);
extern void align_ro(const long dims[DIMS], complex float* odata, const complex float* idata);


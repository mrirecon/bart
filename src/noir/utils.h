
#include <complex.h>

struct linop_s;

extern void noir_calc_weights(double a, double b, const long dims[3], complex float* dst);
extern struct linop_s* linop_noir_weights_create(int N, const long img_dims[N], const long ksp_dims[N], unsigned long flags, double factor_fov, double a, double b, double c);


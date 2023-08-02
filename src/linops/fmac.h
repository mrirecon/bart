
#include <complex.h>

struct linop_s;
extern const struct linop_s* linop_fmac_create(int N, const long dims[N], unsigned long oflags, unsigned long iflags, unsigned long flags, const complex float* tensor);
extern const struct linop_s* linop_fmac_dims_create(int N, const long odims[N], const long idims[N], const long tdims[N], const complex float* tensor);

extern void linop_fmac_set_tensor(const struct linop_s* lop, int N, const long tdims[N], const complex float* tensor);
extern void linop_fmac_set_tensor_F(const struct linop_s* lop, int N, const long tdims[N], const complex float* tensor);
extern void linop_fmac_set_tensor_ref(const struct linop_s* lop, int N, const long tdims[N], const complex float* tensor);
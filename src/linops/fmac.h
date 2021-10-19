
#include <complex.h>

struct linop_s;
extern const struct linop_s* linop_fmac_create(unsigned int N, const long dims[N],
		unsigned int oflags, unsigned int iflags, unsigned int flags, const complex float* tensor);

extern void linop_fmac_set_tensor(const struct linop_s* lop, int N, const long tdims[N], const complex float* tensor);
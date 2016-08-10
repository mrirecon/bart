
#include <complex.h>

extern void nudft_forward2(unsigned int N, unsigned long flags,
			const long odims[N], const long ostrs[N], complex float* out,
			const long idims[N], const long istrs[N], const complex float* in,
			const long tdims[N], const long tstrs[N], const complex float* traj);

extern void nudft_forward(unsigned int N, unsigned long flags,
			const long odims[N], complex float* out,
			const long idims[N], const complex float* in,
			const long tdims[N], const complex float* traj);

extern void nudft_adjoint2(unsigned int N, unsigned long flags,
			const long odims[N], const long ostrs[N], complex float* out,
			const long idims[N], const long istrs[N], const complex float* in,
			const long tdims[N], const long tstrs[N], const complex float* traj);

extern void nudft_adjoint(unsigned int N, unsigned long flags,
			const long odims[N], complex float* out,
			const long idims[N], const complex float* in,
			const long tdims[N], const complex float* traj);

struct linop_s;
extern const struct linop_s* nudft_create2(unsigned int N, unsigned long flags,
					const long odims[N], const long ostrs[N],
					const long idims[N], const long istrs[N],
					const long tdims[N], const complex float* traj);

extern const struct linop_s* nudft_create(unsigned int N, unsigned long flags,
					const long odims[N], const long idims[N],
					const long tdims[N], const complex float* traj);


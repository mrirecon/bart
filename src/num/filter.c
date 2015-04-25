
#include <complex.h>
#include <math.h>
#include <strings.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/loop.h"

#include "misc/misc.h"

#include "filter.h"





void centered_gradient(unsigned int N, const long dims[N], const complex float grad[N], complex float* out)
{
	md_zgradient(N, dims, out, grad);

	long dims0[N];
	md_singleton_dims(N, dims0);

	long strs0[N];
	md_calc_strides(N, strs0, dims0, CFL_SIZE);

	complex float cn = 0.;

	for (unsigned int n = 0; n < N; n++)
		 cn -= grad[n] * (float)dims[n] / 2.;

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	md_zadd2(N, dims, strs, out, strs, out, strs0, &cn);
}

void linear_phase(unsigned int N, const long dims[N], const float pos[N], complex float* out)
{
	complex float grad[N];

	for (unsigned int n = 0; n < N; n++)
		grad[n] = 2.i * M_PI * (float)(pos[n]) / ((float)dims[n]);

	centered_gradient(N, dims, grad, out);
	md_zmap(N, dims, out, out, cexpf);
}


void klaplace(unsigned int N, const long dims[N], unsigned int flags, complex float* out)
{
	unsigned int flags2 = flags;

	complex float* tmp = md_alloc(N, dims, CFL_SIZE);

	md_clear(N, dims, out, CFL_SIZE);

	for (unsigned int i = 0; i < bitcount(flags); i++) {

		unsigned int lsb = ffs(flags2) - 1;
		flags2 = MD_CLEAR(flags2, lsb);

		complex float grad[N];
		for (unsigned int j = 0; j < N; j++)
			grad[j] = 0.;

		grad[lsb] = 1. / (float)dims[lsb];
		centered_gradient(N, dims, grad, tmp);
		md_zspow(N, dims, tmp, tmp, 2.);
		md_zadd(N, dims, out, out, tmp);
	}

	md_free(tmp);
}



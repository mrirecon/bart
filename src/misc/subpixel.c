
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "misc/misc.h"

#include "subpixel.h"



void est_subpixel_shift(unsigned int N, float shifts[N], const long dims[N], unsigned int flags, const complex float* in1, const complex float* in2)
{
	complex float* tmp1 = md_alloc(N, dims, CFL_SIZE);
	complex float* tmp2 = md_alloc(N, dims, CFL_SIZE);

	fftuc(N, dims, flags, tmp1, in1);
	fftuc(N, dims, flags, tmp2, in2);

	md_zmulc(N, dims, tmp1, tmp1, tmp2);

	for (unsigned int i = 0; i < N; i++) {

		shifts[i] = 0.;

		if (!MD_IS_SET(flags, i))
			continue;

		long shift[N];
		for (unsigned int j = 0; j < N; j++)
			shift[j] = 0;

		shift[i] = 1;

		md_circ_shift(N, dims, shift, tmp2, tmp1, CFL_SIZE);

		// the weighting is not optimal due to double squaring
		// and we compute finite differences (filter?)

		complex float sc = md_zscalar(N, dims, tmp2, tmp1);
		shifts[i] = cargf(sc) / (2. * M_PI) * (float)dims[i];
	}

	md_free(tmp1);
	md_free(tmp2);
}



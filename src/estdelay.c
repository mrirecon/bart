/* Copyright 2017. Martin Uecker
 *
 * Kai Tobias Block and Martin Uecker, Simple Method for Adaptive
 * Gradient-Delay Compensation in Radial MRI, Annual Meeting ISMRM,
 * Montreal 2011, In Proc. Intl. Soc. Mag. Reson. Med 19: 2816 (2011)
 *
 * Amir Moussavi, Markus Untenberger, Martin Uecker, and Jens Frahm,
 * Correction of gradient-induced phase errors in radial MRI,
 * Magnetic Resonance in Medicine, 71:308-312 (2014)
 */

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/qform.h"

#include "misc/subpixel.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"

#ifndef DIMS
#define DIMS 16
#endif


static void radial_self_delays(unsigned int N, float shifts[N], const float phi[N], const long dims[DIMS], const complex float* in)
{
	unsigned int d = 2;
	unsigned int flags = (1 << d);

	assert(N == dims[d]);

	long dims1[DIMS];
	md_select_dims(DIMS, ~flags, dims1, dims);

	complex float* tmp1 = md_alloc(DIMS, dims1, CFL_SIZE);
	complex float* tmp2 = md_alloc(DIMS, dims1, CFL_SIZE);

	long pos[DIMS] = { 0 };

	for (unsigned int i = 0; i < dims[d]; i++) {

		pos[d] = i;
		md_copy_block(DIMS, pos, dims1, tmp1, dims, in, CFL_SIZE);

		// find opposing spoke

		float mdelta = 0.;
		int mindex = 0;

		for (unsigned int j = 0; j < dims[d]; j++) {

			float delta = cabsf(cexpf(1.i * phi[j]) - cexpf(1.i * phi[i]));

			if (mdelta <= delta) {

				mdelta = delta;
				mindex = j;
			}
		}

		pos[d] = mindex;
		md_copy_block(DIMS, pos, dims1, tmp2, dims, in, CFL_SIZE);


		unsigned int d2 = 1;
		float rshifts[DIMS];
		md_flip(DIMS, dims1, MD_BIT(d2), tmp2, tmp2, CFL_SIZE); // could be done by iFFT in est_subpixel_shift
		est_subpixel_shift(DIMS, rshifts, dims1, MD_BIT(d2), tmp2, tmp1);

		float mshift = rshifts[d2] / 2.; // mdelta

		shifts[i] = mshift;
	}

	md_free(tmp1);
	md_free(tmp2);
}




static const char usage_str[] = "<trajectory> <data>";
static const char help_str[] = "Estimate gradient delays from radial data.";


int main_estdelay(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	long tdims[DIMS];
	const complex float* traj = load_cfl(argv[1], DIMS, tdims);

	long tdims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc(DIMS, tdims1, CFL_SIZE);
	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ 0 }, tdims, traj1, traj, CFL_SIZE);

	unsigned int N = tdims[2];

	float angles[N];
	for (unsigned int i = 0; i < N; i++)
		angles[i] = M_PI + atan2f(crealf(traj1[3 * i + 0]), crealf(traj1[3 * i + 1]));

	unmap_cfl(DIMS, tdims, traj);


	long dims[DIMS];
	const complex float* in = load_cfl(argv[2], DIMS, dims);

	// FIXME: more checks
	assert(dims[1] == tdims[1]);
	assert(dims[2] == tdims[2]);

	float delays[N];
	radial_self_delays(N, delays, angles, dims, in);


	/* We allow an arbitrary quadratic form to account for
	 * non-physical coordinate systems.
	 * Moussavi et al., MRM 71:308-312 (2014)
	 */
	float qf[3];
	fit_quadratic_form(qf, N, angles, delays);
	printf("%f:%f:%f\n", qf[0], qf[1], qf[2]);


	unmap_cfl(DIMS, dims, in);

	exit(0);
}



/* Copyright 2017-2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 *
 *
 * Kai Tobias Block and Martin Uecker, Simple Method for Adaptive
 * Gradient-Delay Compensation in Radial MRI, Annual Meeting ISMRM,
 * Montreal 2011, In Proc. Intl. Soc. Mag. Reson. Med 19: 2816 (2011)
 *
 * Amir Moussavi, Markus Untenberger, Martin Uecker, and Jens Frahm,
 * Correction of gradient-induced phase errors in radial MRI,
 * Magnetic Resonance in Medicine, 71:308-312 (2014)
 *
 * Sebastian Rosenzweig, Hans Christian Holme, Martin Uecker,
 * Simple Auto-Calibrated Gradient Delay Estimation From Few Spokes Using Radial
 * Intersections (RING), Magnetic Resonance in Medicine 81:1898-1906 (2019)
 */

#include <math.h>

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/mri.h"

#include "num/init.h"
#include "num/qform.h"
#include "num/multind.h"

#include "calib/delays.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif





static const char usage_str[] = "<trajectory> <data>";
static const char help_str[] = "Estimate gradient delays from radial data.";


int main_estdelay(int argc, char* argv[])
{
	bool do_ring = false;
	struct ring_conf conf = ring_defaults;

	const struct opt_s opts[] = {

		OPT_SET('R', &do_ring, "RING method"),
		OPT_UINT('p', &conf.pad_factor, "p", "[RING] Padding"),
		OPT_UINT('n', &conf.no_intersec_sp, "n", "[RING] Number of intersecting spokes"),
		OPT_FLOAT('r', &conf.size, "r", "[RING] Central region size"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (0 != conf.pad_factor % 2)
		error("Pad_factor -p should be even\n");


	long tdims[DIMS];
	const complex float* traj = load_cfl(argv[1], DIMS, tdims);

	long tdims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc(DIMS, tdims1, CFL_SIZE);
	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ 0 }, tdims, traj1, traj, CFL_SIZE);

	int N = tdims[2];

	float angles[N];
	for (int i = 0; i < N; i++)
		angles[i] = M_PI + atan2f(crealf(traj1[3 * i + 0]), crealf(traj1[3 * i + 1]));


	if (do_ring) {

		assert(0 == tdims[1] % 2);

		md_slice(DIMS, MD_BIT(1), (long[DIMS]){ [1] = tdims[1] / 2 }, tdims, traj1, traj, CFL_SIZE);

		for (int i = 0; i < N; i++)
			if (0. != cabsf(traj1[3 * i]))
				error("Nominal trajectory must be centered for RING.\n");
	}


	md_free(traj1);


	long full_dims[DIMS];
	const complex float* full_in = load_cfl(argv[2], DIMS, full_dims);

	// Remove not needed dimensions
	long dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|PHS1_FLAG|PHS2_FLAG|COIL_FLAG, dims, full_dims);

	complex float* in = md_alloc(DIMS, dims, CFL_SIZE);

	long pos[DIMS] = { 0 };
	md_copy_block(DIMS, pos, dims, in, full_dims, full_in, CFL_SIZE);

	// FIXME: more checks
	assert(dims[1] == tdims[1]);
	assert(dims[2] == tdims[2]);

	float qf[3];	// S in RING

	if (!do_ring) {

		// Block and Uecker, ISMRM 19:2816 (2001)

		float delays[N];
		radial_self_delays(N, delays, angles, dims, in);

		/* We allow an arbitrary quadratic form to account for
		 * non-physical coordinate systems.
		 * Moussavi et al., MRM 71:308-312 (2014)
		 */

		fit_quadratic_form(qf, N, angles, delays);

	} else {

		/* RING method
		 * Rosenzweig et al., MRM 81:1898-1906 (2019)
		 */

		ring(&conf, qf, N, angles, dims, in);
	}

	bart_printf("%f:%f:%f\n", qf[0], qf[1], qf[2]);

	unmap_cfl(DIMS, full_dims, full_in);
	unmap_cfl(DIMS, tdims, traj);
	md_free(in);

	return 0;
}



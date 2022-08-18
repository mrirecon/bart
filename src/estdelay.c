/* Copyright 2017-2020. Uecker Lab. University Medical Center Göttingen.
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
#include "misc/debug.h"

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





static const char help_str[] = "Estimate gradient delays from radial data.";


int main_estdelay(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* data_file = NULL;
	const char* qf_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "trajectory"),
		ARG_INFILE(true, &data_file, "data"),
		ARG_OUTFILE(false, &qf_file, "qf"),
	};

	bool do_ring = false;
	struct ring_conf conf = ring_defaults;

	const struct opt_s opts[] = {

		OPT_SET('R', &do_ring, "RING method"),
		OPT_UINT('p', &conf.pad_factor, "p", "[RING] Padding"),
		OPT_UINT('n', &conf.no_intersec_sp, "n", "[RING] Number of intersecting spokes"),
		OPT_FLOAT('r', &conf.size, "r", "[RING] Central region size"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (0 != conf.pad_factor % 2)
		error("Pad_factor -p should be even\n");


	long tdims[DIMS];
	const complex float* traj = load_cfl(traj_file, DIMS, tdims);


	int N = tdims[2];

	float angles[N];

	traj_radial_angles(N, angles, tdims, traj);

	float dc_shift = traj_radial_dcshift(tdims, traj);
	float scale = traj_radial_dk(tdims, traj);

	// Warn on unexpected shifts: != 0.5 for even number of samples, != 0 for odd number of sampled
	if (1 == tdims[1] % 2) {

		debug_printf(DP_WARN, "odd number of samples\n");
		if (fabsf(dc_shift/scale - 0.0f) > 0.0001)
			debug_printf(DP_WARN, "DC is shifted by: %f [sample], 1 sample = %f [1/FOV]\n", dc_shift, scale);

	} else if (fabsf(dc_shift/scale - 0.5f) > 0.0001) {

		debug_printf(DP_WARN, "DC is shifted by: %f [sample], 1 sample = %f [1/FOV]\n", dc_shift, scale);
	}


	long full_dims[DIMS];
	const complex float* full_in = load_cfl(data_file, DIMS, full_dims);

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

		if (0 == tdims[1] % 2) {

			qf[0] += 0.5;
			qf[1] += 0.5;
		}

	} else {

		/* RING method
		 * Rosenzweig et al., MRM 81:1898-1906 (2019)
		 */

		ring(&conf, qf, N, angles, dims, in);
	}

	qf[0] -= dc_shift / scale;
	qf[1] -= dc_shift / scale;


	bart_printf("%f:%f:%f\n", qf[0], qf[1], qf[2]);

	if (NULL != qf_file) {

		long qf_dims[1] = { 3 };

		complex float* oqf = create_cfl(qf_file, 1, qf_dims);

		for (int i = 0; i < 3; i++)
			oqf[i] = qf[i];

		unmap_cfl(1, qf_dims, oqf);
	}

	unmap_cfl(DIMS, full_dims, full_in);
	unmap_cfl(DIMS, tdims, traj);
	md_free(in);

	return 0;
}



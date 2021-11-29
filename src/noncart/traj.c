/* Copyright 2014-2015 The Regents of the University of California.
 * Copyright 2015-2020 Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2014-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2020 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"
#ifdef SSAFARY_PAPER
#include "misc/debug.h"
#endif

#include "traj.h"


const struct traj_conf traj_defaults = {

	.radial = false,
	.golden = false,
	.aligned = false,
	.full_circle = false,
	.half_circle_gold = false,
	.golden_partition = false,
	.d3d = false,
	.transverse = false,
	.asym_traj = false,
	.mems_traj = false,
	.accel = 1,
	.tiny_gold = 0,
};


void euler(float dir[3], float phi, float psi)
{
	dir[0] = cosf(phi) * cosf(psi);
	dir[1] = sinf(phi) * cosf(psi);
	dir[2] =             sinf(psi);
}


/* We allow an arbitrary quadratic form to account for
 * non-physical coordinate systems.
 * Moussavi et al., MRM 71:308-312 (2014)
 */
void gradient_delay(float d[3], float coeff[2][3], float phi, float psi)
{
	float dir[3];
	euler(dir, phi, psi);

	float mat[3][3] = {

		{ coeff[0][0], coeff[0][2], coeff[1][1] },
		{ coeff[0][2], coeff[0][1], coeff[1][2] },
		{ coeff[1][1], coeff[1][2], coeff[1][0] },
	};

	for (unsigned int i = 0; i < 3; i++) {

		d[i] = 0.;

		for (unsigned int j = 0; j < 3; j++)
			d[i] += mat[i][j] * dir[j];
	}
}

void calc_base_angles(double base_angle[DIMS], int Y, int E, int mb, int turns, struct traj_conf conf)
{
	/*
	 * Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O.
	 * An optimal radial profile order based on the Golden Ratio
	 * for time-resolved MRI. IEEE TMI 26:68--76 (2007)
	 *
	 * Wundrak S, Paul J, Ulrici J, Hell E, Geibel MA, Bernhardt P, Rottbauer W, Rasche V.
	 * Golden ratio sparse MRI using tiny golden angles.
	 * Magn Reson Med 75:2372-2378 (2016)
	 */

	double golden_ratio = (sqrt(5.) + 1.) / 2;
	if (use_compat_to_version("v0.5.00"))
		golden_ratio = (sqrtf(5.) + 1.) / 2;

	double golden_angle = M_PI / (golden_ratio + conf.tiny_gold - 1.);

	// For numerical stability
	if (1 == conf.tiny_gold) {

		golden_angle = M_PI * (2. - (3. - sqrt(5.))) / 2.;
		if (use_compat_to_version("v0.5.00"))
			golden_angle = M_PI * (2. - (3. - sqrtf(5.))) / 2.;
	}



	double angle_atom = M_PI / Y;

	// Angle between spokes of one slice/partition
	double angle_s = angle_atom * (conf.full_circle ? 2 : 1);

	// Angle between slices/partitions
	double angle_m = angle_atom / mb; // linear-turned partitions

	if (conf.aligned)
		angle_m = 0;

	// Angle between turns
	double angle_t = 0.;

	if (turns > 1)
		angle_t = angle_atom / turns * (conf.full_circle ? 2 : 1);

	/* radial multi-echo multi-spoke sampling
	 *
	 * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
	 * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
	 * estimation using undersampled  triple-echo multi-spoke radial FLASH.
	 * Magn Reson Med 82:1000-1011 (2019)
	 */
	double angle_e = 0.;

	if (conf.mems_traj) {

		angle_s = angle_s * 1.;
		angle_e = angle_s / E;
		angle_t = golden_angle;

	} else if (conf.golden) {

		if (conf.aligned) {

			angle_s = golden_angle;
			angle_m = 0;
			angle_t = golden_angle * Y;

		} else {

			angle_s = golden_angle;
			angle_m = golden_angle * Y;
			angle_t = golden_angle * Y * mb;
		}

#ifdef SSAFARY_PAPER
		/* Specific trajectory designed for z-undersampled Stack-of-Stars imaging:
		 *
		 * Sebastian Rosenzweig, Nick Scholand, H. Christian M. Holme, Martin Uecker.
		 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular Spectrum
		 * Analysis (SSA-FARY). IEEE Tran Med Imag 2020; 10.1109/TMI.2020.2985994. arXiv:1812.09057.
		 */

		if (14 == mb) {

			int mb_red = 8;

			angle_m = golden_angle;
			angle_s = golden_angle * mb_red;
			angle_t = golden_angle * Y * mb_red;

			debug_printf(DP_INFO, "Trajectory generation to reproduce SSA-FARY Paper!\n");
		}
#endif
		if (use_compat_to_version("v0.4.00")) {

			// since the traj rewrite (commit d4e6e2e3a2313) we do not apply
			// full circle to golden angle anymore. However, this is needed for
			// reproducing the RING paper
			angle_s *= (conf.full_circle ? 2. : 1.);
			angle_m *= (conf.full_circle ? 2. : 1.);
			angle_t *= (conf.full_circle ? 2. : 1.);
		}

	}

	base_angle[PHS2_DIM] = angle_s;
	base_angle[SLICE_DIM] = angle_m;
	base_angle[TE_DIM] = angle_e;
	base_angle[TIME_DIM] = angle_t;
}


// z-Undersampling
bool zpartition_skip(long partitions, long z_usamp[2], long partition, long frame)
{
	long z_reflines = z_usamp[0];
	long z_acc = z_usamp[1];

	if (1 == z_acc) // No undersampling. Do not skip partition
		return false;


	// Auto-Calibration region

	long DC_idx = partitions / 2;
	long AC_lowidx = DC_idx - floor(z_reflines / 2.);
	long AC_highidx = DC_idx + ceil(z_reflines / 2.) - 1;

	if ((partition >= AC_lowidx) && (partition <= AC_highidx)) // Auto-calibration line. Do not skip partition.
		return false;

	// Check if this non-Auto-calibration line should be sampled.

	long part = (partition < AC_lowidx) ? partition : (partition - AC_highidx - 1);

	if (0 == ((part - (frame % z_acc)) % z_acc))
		return false;

	return true;
}


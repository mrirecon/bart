/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2014-2020 Martin Uecker
 * 2018-2020 Sebastian Rosenzweig
 * 2019-2020 Zhengguo Tan
 */

#include <stdbool.h>
#include <math.h>
#include <assert.h>

#include "num/multind.h"

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
	.mems_legacy = false,
	.accel = 1,
	.tiny_gold = 0,
	.rational = false,
	.double_base = false,
	.turns = 1,
	.mb = 1,
	.Y = 1,
	.raga_inc = 1,
	.aligned_flags = 0,
};


static void euler(float dir[3], float phi, float psi)
{
	dir[0] = cosf(phi) * cosf(psi);
	dir[1] = sinf(phi) * cosf(psi);
	dir[2] =             sinf(psi);
}


void traj_read_dir(float dir[3], float phi, float psi)
{
	euler(dir, phi, psi);

	SWAP(dir[0], dir[1]);
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

	for (int i = 0; i < 3; i++) {

		d[i] = 0.;

		for (int j = 0; j < 3; j++)
			d[i] += mat[i][j] * dir[j];
	}

	SWAP(d[0], d[1]);
}

int recover_gen_fib_ind(int Y, int inc)
{
	int step = 0;

	while (inc > gen_fibonacci(1, step - 1))
		step++;

	// Assumption: Fib Index < 40
	for (int i = 1; i <= 40; i++)
		if (Y == gen_fibonacci(i, step))
			return i;

	return -1;
}

static void fib_next(int f[2])
{
	int t = f[0];
	f[0] = f[1];
	f[1] += t;
}

int gen_fibonacci(int n, int ind)
{
	int fib[2] = { 1, n };

	for (int i = 0; i < ind - 1; i++)
		fib_next(fib);

	return (0 == ind) ? fib[0] : fib[1];
}

int raga_find_index(int Y, int n)
{
	int i = 0;

	while (Y > gen_fibonacci(n, i))
		i++;

	return i;
}

int raga_increment(int Y, int n)
{
	int i = raga_find_index(Y, n);

	return gen_fibonacci(1, i - 1);
}

int raga_spokes(int baseresolution, int tiny_ga)
{
	int i = raga_find_index((M_PI / 2.) * baseresolution, tiny_ga);

	while (0 == gen_fibonacci(tiny_ga, i) % 2)
		i--;

	return gen_fibonacci(tiny_ga, i);
}

static double calc_golden_angle(int tiny_gold)
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

	double golden_angle = M_PI / (golden_ratio + tiny_gold - 1.);

	// For numerical stability
	if (1 == tiny_gold) {

		golden_angle = M_PI * (2. - (3. - sqrt(5.))) / 2.;

		if (use_compat_to_version("v0.5.00"))
			golden_angle = M_PI * (2. - (3. - sqrtf(5.))) / 2.;
	}

	return golden_angle;
}

double calc_angle_atom(const struct traj_conf* conf)
{
	assert(conf->rational);
	return 2. * M_PI / (double)conf->Y;
}

void calc_base_angles(double base_angle[DIMS], int Y, int E, struct traj_conf conf)
{
	assert(!conf.rational);

	double angle_atom = M_PI / Y;
	double golden_angle = calc_golden_angle(conf.tiny_gold);


	if (conf.double_base)
		golden_angle *= 2.;

	double angle_s = 0.;
	double angle_m = 0.; // slices/partitions
	double angle_t = 0.; // turns
	double angle_e = 0.;

	if (!conf.golden || conf.mems_traj) {

		// Angle between spokes of one slice/partition
		angle_s = angle_atom * (conf.full_circle ? 2 : 1);

		if (!conf.aligned)
			angle_m = angle_atom / conf.mb; // linear-turned partitions

		if (conf.turns > 1)
			angle_t = angle_atom / conf.turns * (conf.full_circle ? 2 : 1);

		if (conf.mems_traj) {

			/* radial multi-echo multi-spoke sampling
			 *
			 * Tan Z, Voit D, Kollmeier JM, Uecker M, Frahm J.
			 * Dynamic water/fat separation and B0 inhomogeneity mapping -- joint
			 * estimation using undersampled  triple-echo multi-spoke radial FLASH.
			 * Magn Reson Med 82:1000-1011 (2019)
			 */

			angle_e = angle_s / E + M_PI;
			if (conf.mems_legacy)
				angle_e = angle_s / E;
			angle_t = golden_angle;
		}

	} else {

		if (conf.aligned) {

			angle_s = golden_angle;
			angle_m = 0;
			angle_t = golden_angle * Y;

		} else {

#if 0			
			// FIXME 
			// fix tests/test-traj-rational-approx-multislice; mrirecon/sms-t1-mapping)
			angle_s = golden_angle * conf.mb;
			angle_m = golden_angle;
#endif
			angle_s = golden_angle;
			angle_m = golden_angle * Y;
			angle_t = golden_angle * Y * conf.mb;
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

			angle_s = golden_angle * mb_red;
			angle_m = golden_angle;
			angle_t = golden_angle * Y * mb_red;

			debug_printf(DP_INFO, "Trajectory generation to reproduce SSA-FARY Paper!\n");
		}
#endif
		if (use_compat_to_version("v0.4.00") && conf.full_circle) {

			// since the traj rewrite (commit d4e6e2e3a2313) we do not apply
			// full circle to golden angle anymore. However, this is needed for
			// reproducing the RING paper

			angle_s *= 2.;
			angle_m *= 2.;
			angle_t *= 2.;
		}
	}

	base_angle[PHS2_DIM] = angle_s;
	base_angle[SLICE_DIM] = angle_m;
	base_angle[TE_DIM] = angle_e;
	base_angle[TIME_DIM] = angle_t;
}


long raga_increment_from_pos(const int order[DIMS], const long pos[DIMS], unsigned long flags, const long dims[DIMS], const struct traj_conf* conf)
{
	assert(conf->rational);

	unsigned long outer_loops = 0;

	bool outer = false;
	for (int d = 0; d < DIMS; d++) {

		if (outer)
			outer_loops |= (1UL << order[d]);

		if (TIME_DIM == order[d])
			outer = true;
	}

	long idx_inner = md_ravel_index_permuted(DIMS, pos, flags & ~(conf->aligned_flags | outer_loops), dims, order);
	long idx_outer = md_ravel_index_permuted(DIMS, pos, flags & ~conf->aligned_flags & outer_loops, dims, order);

	return conf->raga_inc * (idx_inner + idx_outer) % conf->Y;
}



void indices_from_position(long ind[DIMS], const long pos[DIMS], struct traj_conf conf)
{
	assert(!conf.rational);

	ind[PHS2_DIM] = pos[PHS2_DIM];
	ind[SLICE_DIM] = pos[SLICE_DIM];
	ind[TE_DIM] = pos[TE_DIM];
	ind[TIME_DIM] = pos[TIME_DIM];

	if (conf.turns > 1)
		ind[TIME_DIM] = pos[TIME_DIM] % conf.turns;
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


/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"


static const char usage_str[] = "<output>";
static const char help_str[] = "Computes k-space trajectories.";


static void euler(float dir[3], float phi, float psi)
{
	dir[0] = cosf(phi) * cosf(psi);
	dir[1] = sinf(phi) * cosf(psi);
	dir[2] =             sinf(psi);
}


/* We allow an arbitrary quadratic form to account for
 * non-physical coordinate systems.
 * Moussavi et al., MRM 71:308-312 (2014)
 */
static void gradient_delay(float d[3], float coeff[2][3], float phi, float psi)
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

enum part_mode { REGULAR, LINEAR, ALIGNED };

static int remap(enum part_mode mode, int all, int turns, int mb, int n)
{
	int spp = all / (turns * mb);
	int spt = all / turns;
//	int ind_sp = ((n % (spp * mb)) % spp) * mb * turns;
	int ind_sp = (n % spp) * mb * turns;
	int ind_fr = ((n % (turns * spp)) / spp) * mb;
//	int ind_pr = (n / (turns * spp)) * turns;

	switch (mode) {
	case REGULAR:
		return (n % spt) * turns + n / spt;
	case LINEAR:
		return ind_sp + ind_fr;// + ind_pr;
	case ALIGNED:
		return ind_sp + ind_fr;
	}
	assert(0);
}

int main_traj(int argc, char* argv[])
{
	int X = 128;
	int Y = 128;
	int mb = 0;
	int accel = 1;
	bool radial = false;
	bool golden = false;
	bool aligned = false;
	bool dbl = false;
	bool pGold = false;
	int turns = 1;
	bool d3d = false;
	bool transverse = false;
	bool asymTraj = false;
	bool halfCircle = false;

	float gdelays[2][3] = {
		{ 0., 0., 0. },
		{ 0., 0., 0. }
	};

	const struct opt_s opts[] = {

		OPT_INT('x', &X, "x", "readout samples"),
		OPT_INT('y', &Y, "y", "phase encoding lines"),
		OPT_INT('a', &accel, "a", "acceleration"),
		OPT_INT('t', &turns, "t", "turns"),
		OPT_INT('m', &mb, "mb", "SMS multiband factor"),
		OPT_SET('l', &aligned, "aligned partition angle"),
		OPT_SET('g', &pGold, "golden angle in partition direction"),
		OPT_SET('r', &radial, "radial"),
		OPT_SET('G', &golden, "golden-ratio sampling"),
		OPT_SET('H', &halfCircle, "halfCircle golden-ratio sampling"),
		OPT_SET('D', &dbl, "double base angle"),
		OPT_FLVEC3('q', &gdelays[0], "delays", "gradient delays: x, y, xy"),
		OPT_FLVEC3('Q', &gdelays[1], "delays", "(gradient delays: z, xz, yz)"),
		OPT_SET('O', &transverse, "correct transverse gradient error for radial tajectories"),
		OPT_SET('3', &d3d, "3D"),
		OPT_SET('c', &asymTraj, "Asymmetric trajectory [DC sampled]"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int spp = Y;		// spokes per partition

	if (0 != mb)
		Y = Y * mb * turns;	// total number of spokes

	int N = X * Y / accel;
	long dims[DIMS] = { [0 ... DIMS - 1] = 1  };
	dims[0] = 3;
	dims[1] = X;

	if (halfCircle)
		golden = true;

	if (0 == mb) {

		mb = 1;

	} else {

		dims[TIME_DIM] = turns;
		dims[SLICE_DIM] = mb;
	}


	enum part_mode mode = LINEAR;

	if (golden) {

		radial = true;

		if ((turns != 1) || (mb != 1))
			error("No turns and SMS implemented for golden angle!");

	} else if (dbl || radial) {

		radial = true;

		if (d3d)
			error("3D radial trajectory not implemented yet!");

		if ((mb != 1) && (turns != 1))
			if (0 == turns % mb)
				error("'turns % multiband factor' must be nonzero!");

		if (aligned || pGold)
			mode = ALIGNED;

	} else {

		if ((turns != 1) || (mb != 1))
			error("No turns or spokes in Cartesian trajectories please!");
	}

	dims[2] = (radial ? spp : (Y / accel));

	complex float* samples = create_cfl(argv[1], DIMS, dims);

	int p = 0;
	for (int j = 0; j < Y; j += accel) {
		for (int i = 0; i < X; i++) {

			if (radial) {

				/* golden-ratio sampling
				 *
				 * Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O.
				 * An optimal radial profile order based on the Golden Ratio
				 * for time-resolved MRI. IEEE TMI 26:68--76 (2007)
				 */

				double golden_angle = 3. - sqrtf(5.);
				double base = golden ? ((2. - golden_angle) / 2.) : (1. / (float)Y);
				double angle = M_PI * (float)remap(mode, Y, turns, mb, j) * (dbl ? 2. : 1.) * base;

				if (halfCircle)
					angle = fmod(angle, M_PI);

				/* Calculate read-out samples
				* for symmetric Trajectory [DC between between sample no. X/2-1 and X/2, zero-based indexing]
				* or asymmetric Trajectory [DC component at sample no. X/2, zero-based indexing]
				*/
				double read = (float)i + (asymTraj ? 0 : 0.5) - (float)X / 2.;

				double angle2 = 0.;

				if (d3d) {

					int split = sqrtf(Y);
					angle2 = 2. * M_PI * j * split * base;
				}


				if (!(aligned || pGold)) {

					int pt_ind = j / (turns * spp);
					double angle_part = M_PI / (float)Y * turns;
					angle += pt_ind * angle_part;
				}

				if (pGold) {

					int part = (int)((j % (spp * mb)) / spp); // current partition
					angle += fmod(part * M_PI / spp * (sqrt(5.) - 1) / 2, M_PI / spp);
				}

				float d[3] = { 0., 0., 0 };
				gradient_delay(d, gdelays, angle, angle2);

				float read_dir[3];
				euler(read_dir, angle, angle2);

				if (!transverse) {

					// project to read direction

					float delay = 0.;

					for (unsigned int i = 0; i < 3; i++)
						delay += read_dir[i] * d[i];

					for (unsigned int i = 0; i < 3; i++)
						d[i] = delay * read_dir[i];
				}

				samples[p * 3 + 0] = d[1] + read * read_dir[1];
				samples[p * 3 + 1] = d[0] + read * read_dir[0];
				samples[p * 3 + 2] = d[2] + read * read_dir[2];

			} else {

				samples[p * 3 + 0] = (i - X / 2);
				samples[p * 3 + 1] = (j - Y / 2);
				samples[p * 3 + 2] = 0;
			}

			p++;
		}
	}
	assert(p == N - 0);

	unmap_cfl(3, dims, samples);
	return 0;
}



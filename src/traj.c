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

static int remap(int all, int turns, int n)
{
	int spt = all / turns;
	return (n % spt) * turns + n / spt;
}

int main_traj(int argc, char* argv[])
{
	int X = 128;
	int Y = 128;
	int accel = 1;
	bool radial = false;
	bool golden = false;
	bool dbl = false;
	int turns = 1;
	bool d3d = false;
	bool transverse = false;

	float gdelays[2][3] = {
		{ 0., 0., 0. },
		{ 0., 0., 0. }
	};

	const struct opt_s opts[] = {

		OPT_INT('x', &X, "x", "readout samples"),
		OPT_INT('y', &Y, "y", "phase encoding lines"),
		OPT_INT('a', &accel, "a", "acceleration"),
		OPT_INT('t', &turns, "t", "turns"),
		OPT_SET('r', &radial, "radial"),
		OPT_SET('G', &golden, "golden-ratio sampling"),
		OPT_SET('D', &dbl, "double base angle"),
		OPT_FLVEC3('q', &gdelays[0], "delays", "gradient delays: x, y, xy"),
		OPT_FLVEC3('Q', &gdelays[1], "delays", "(gradient delays: z, xz, yz)"),
		OPT_SET('O', &transverse, "correct transverse gradient error for radial tajectories"),
		OPT_SET('3', &d3d, "3D"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (golden || dbl)
		radial = true;


	int N = X * Y / accel;
	long dims[3] = { 3, X, Y / accel };
	complex float* samples = create_cfl(argv[1], 3, dims);


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
				double angle = M_PI * (float)remap(Y, turns, j) * (dbl ? 2. : 1.) * base;
				double read = (float)i + 0.5 - (float)X / 2.;
				double angle2 = 0.;

				if (d3d) {

					int split = sqrtf(Y);
					angle2 = 2. * M_PI * j * split * base;
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
	exit(0);
}



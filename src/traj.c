/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"


static const char* usage_str = "<output>";
static const char* help_str = "Computes k-space trajectories.";



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

	const struct opt_s opts[] = {

		OPT_INT('x', &X, "x", "readout samples"),
		OPT_INT('y', &Y, "y", "phase encoding lines"),
		OPT_INT('a', &accel, "a", "acceleration"),
		OPT_INT('t', &turns, "t", "turns"),
		OPT_SET('r', &radial, "radial"),
		OPT_SET('G', &golden, "golden-ratio sampling"),
		OPT_SET('D', &dbl, "double base angle"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

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

				samples[p * 3 + 0] = ((float)i + 0.5 - (float)X / 2.) * sin(angle);
				samples[p * 3 + 1] = ((float)i + 0.5 - (float)X / 2.) * cos(angle);
				samples[p * 3 + 2] = 0.;

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



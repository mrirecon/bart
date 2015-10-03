/* Copyright 2014-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012, 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <getopt.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-h] [-r] <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Computes k-space trajectories.\n"
		"\n"
		"-x x\treadout samples\n"
		"-y y\tphase encoding lines\n"
		"-a a\tacceleration\n"
		"-t t\tturns\n"
		"-r\tradial\n"
		"-G\tgolden-ratio sampling\n"
		"-D\tdouble base angle\n"
		"-h\thelp\n");
}


static int remap(int all, int turns, int n)
{
	int spt = all / turns;
	return (n % spt) * turns + n / spt;
}

int main(int argc, char* argv[])
{
	int X = 128;
	int Y = 128;
	int accel = 1;
	bool radial = false;
	bool golden = false;
	bool dbl = false;
	int c;
	int turns = 1;

	while (-1 != (c = getopt(argc, argv, "x:y:a:t:rDGh"))) {

		switch (c) {

		case 'x':
			X = atoi(optarg);
			break;

		case 'y':
			Y = atoi(optarg);
			break;

		case 'a':
			accel = atoi(optarg);
			break;

		case 'G':
			golden = true;
			radial = true;
			break;

		case 'D':
			dbl = true;
		case 'r':
			radial = true;
			break;

		case 't':
			turns = atoi(optarg);
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 1) {

		usage(argv[0], stderr);
		exit(1);
	}

	int N = X * Y / accel;
	long dims[3] = { 3, X, Y / accel };
	complex float* samples = create_cfl(argv[optind + 0], 3, dims);


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



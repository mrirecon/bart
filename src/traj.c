/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2019 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2019 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019 Aurélien Trotier <a.trotier@gmail.com>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include <stdint.h>

#include "num/flpmath.h"

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noncart/traj.h"

static const char usage_str[] = "<output>";
static const char help_str[] = "Computes k-space trajectories.";


int main_traj(int argc, char* argv[])
{
	int X = 128;
	int Y = 128;
	int mb = 1;
	int turns = 1;
	float rot = 0.;

	struct traj_conf conf = traj_defaults;

	float gdelays[2][3] = {
		{ 0., 0., 0. },
		{ 0., 0., 0. }
	};

	long z_usamp[2] = { 0, 1 }; // { reference Lines, acceleration }

	const char* custom_angle = NULL;


	const struct opt_s opts[] = {

		OPT_INT('x', &X, "x", "readout samples"),
		OPT_INT('y', &Y, "y", "phase encoding lines"),
		OPT_INT('a', &conf.accel, "a", "acceleration"),
		OPT_INT('t', &turns, "t", "turns"),
		OPT_INT('m', &mb, "mb", "SMS multiband factor"),
		OPT_SET('l', &conf.aligned, "aligned partition angle"),
		OPT_SET('g', &conf.golden_partition, "golden angle in partition direction"),
		OPT_SET('r', &conf.radial, "radial"),
		OPT_SET('G', &conf.golden, "golden-ratio sampling"),
		OPT_SET('H', &conf.half_circle_gold, "halfCircle golden-ratio sampling"),
		OPT_INT('s', &conf.tiny_gold, "# Tiny GA", "tiny golden angle"),
		OPT_SET('D', &conf.full_circle, "projection angle in [0,360°), else in [0,180°)"),
		OPT_FLOAT('R', &rot, "phi", "rotate"),
		OPT_FLVEC3('q', &gdelays[0], "delays", "gradient delays: x, y, xy"),
		OPT_FLVEC3('Q', &gdelays[1], "delays", "(gradient delays: z, xz, yz)"),
		OPT_SET('O', &conf.transverse, "correct transverse gradient error for radial tajectories"),
		OPT_SET('3', &conf.d3d, "3D"),
		OPT_SET('c', &conf.asym_traj, "asymmetric trajectory [DC sampled]"),
		OPT_VEC2('z', &z_usamp, "Ref:Acel", "Undersampling in z-direction."),
		OPT_STRING('C', &custom_angle, "file", "custom_angle"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	// Load custom_angle
	long sdims[DIMS];
	complex float* custom_angle_val = NULL;

	if (NULL != custom_angle && conf.radial) {

		debug_printf(DP_INFO, "custom_angle file is used \n");
		custom_angle_val = load_cfl(custom_angle, DIMS, sdims);

		if(Y != sdims[0]){

			debug_printf(DP_INFO, "According to the custom angle file : y = %d\n",sdims[0]);
			Y = sdims[0];

		}
	}

	int tot_sp = Y * mb * turns;	// total number of lines/spokes
	int N = X * tot_sp / conf.accel;


	long dims[DIMS] = { [0 ... DIMS - 1] = 1  };
	dims[0] = 3;
	dims[1] = X;
	dims[2] = (conf.radial ? Y : (Y / conf.accel));

	// Variables for z-undersampling
	long z_reflines = z_usamp[0];
	long z_acc = z_usamp[1];

	long mb2 = mb;

	if (z_acc > 1) {

		mb2 = z_reflines + (mb - z_reflines) / z_acc;

		if ((mb2 < 1) || ((mb - z_reflines) % z_acc != 0))
			error("Invalid z-Acceleration!\n");

	}


	dims[TIME_DIM] = turns;
	dims[SLICE_DIM] = mb;

	if (conf.half_circle_gold) {

		conf.golden = true;

		if (conf.full_circle)
			error("Invalid options. Full-circle or half-circle sampling?");
	}


	if (conf.d3d) {

		if (turns >1)
			error("Turns not implemented for 3D-Kooshball\n");

		if (mb > 1)
			error("Multiple partitions not sensible for 3D-Kooshball\n");
	}

	if (conf.tiny_gold >= 1)
		conf.golden = true;

	if (conf.golden) {

		conf.radial = true;

		if (0 == conf.tiny_gold)
			conf.tiny_gold = 1;

	} else if (conf.full_circle || conf.radial) {

		conf.radial = true;

	} else { // Cartesian

		if ((turns != 1) || (mb != 1))
			error("Turns or partitions not allowed/implemented for Cartesian trajectories!");
	}


	complex float* samples = create_cfl(argv[1], DIMS, dims);

	md_clear(DIMS, dims, samples, CFL_SIZE);

	double golden_ratio = (sqrtf(5.) + 1.) / 2;
	double angle_atom = M_PI / Y;

	double base_angle[DIMS] = { 0. };
	calc_base_angles(base_angle, Y, mb2, turns, conf);

	int p = 0;
	long pos[DIMS] = { 0 };

	do {

		int i = pos[PHS1_DIM];
		int j = pos[PHS2_DIM];
		int m = pos[SLICE_DIM];

		if (conf.radial) {

			int s = j;

			/* Calculate read-out samples
			 * for symmetric trajectory [DC between between sample no. X/2-1 and X/2, zero-based indexing]
			 * or asymmetric trajectory [DC component at sample no. X/2, zero-based indexing]
			 */
			double read = (float)i + (conf.asym_traj ? 0 : 0.5) - (float)X / 2.;

			if (conf.golden_partition)
				base_angle[1] = (m > 0) ? (fmod(angle_atom * m / golden_ratio, angle_atom) / m) : 0;

			double angle = 0.;

			for (unsigned int d = 1; d < DIMS; d++)
				angle += pos[d] * base_angle[d];


			if (conf.half_circle_gold)
				angle = fmod(angle, M_PI);

			angle += M_PI * rot / 180.;

			// 3D
			double angle2 = 0.;

			if (conf.d3d) {

				int split = sqrtf(Y);
				angle2 = s * M_PI / Y * (conf.full_circle ? 2 : 1) * split;

				if (NULL != custom_angle)
						angle2 = cimag(custom_angle_val[p%X]);

			}


			if (NULL != custom_angle)
					angle = creal(custom_angle_val[p%X]);


			float d[3] = { 0., 0., 0 };
			gradient_delay(d, gdelays, angle, angle2);

			float read_dir[3];
			euler(read_dir, angle, angle2);

			if (!conf.transverse) {

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

	} while(md_next(DIMS, dims, ~1L, pos));

	assert(p == N - 0);

	if (NULL != custom_angle_val)
		unmap_cfl(3, sdims, custom_angle_val);

	unmap_cfl(3, dims, samples);

	exit(0);
}




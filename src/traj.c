/* Copyright 2014-2015. The Regents of the University of California.
 * Copyright 2015-2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018-2019 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 * 2019 Aurélien Trotier <a.trotier@gmail.com>
 * 2019-2020 Zhengguo Tan <zhengguo.tan@med.uni-goettingen.de>
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

static const char help_str[] = "Computes k-space trajectories.";


int main_traj(int argc, char* argv[argc])
{
	const char* out_file= NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "output"),
	};

	int X = 128;
	int Y = 128;
	int D = -1;
	int E = 1;
	int mb = 1;
	int turns = 1;
	float rot = 0.;
	float over = 1.;

	struct traj_conf conf = traj_defaults;

	float gdelays[2][3] = {
		{ 0., 0., 0. },
		{ 0., 0., 0. }
	};

	long z_usamp[2] = { 0, 1 }; // { reference Lines, acceleration }

	const char* custom_angle_file = NULL;
	const char* gdelays_file = NULL;


	const struct opt_s opts[] = {

		OPT_INT('x', &X, "x", "readout samples"),
		OPT_INT('y', &Y, "y", "phase encoding lines"),
		OPT_INT('d', &D, "d", "full readout samples"),
		OPT_INT('e', &E, "e", "number of echoes"),
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
		OPT_FLOAT('o', &over, "o", "oversampling factor"),
		OPT_FLOAT('R', &rot, "phi", "rotate"),
		OPT_FLVEC3('q', &gdelays[0], "delays", "gradient delays: x, y, xy"),
		OPT_FLVEC3('Q', &gdelays[1], "delays", "(gradient delays: z, xz, yz)"),
		OPT_SET('O', &conf.transverse, "correct transverse gradient error for radial tajectories"),
		OPT_SET('3', &conf.d3d, "3D"),
		OPT_SET('c', &conf.asym_traj, "asymmetric trajectory [DC sampled]"),
		OPT_SET('E', &conf.mems_traj, "multi-echo multi-spoke trajectory"),
		OPT_VEC2('z', &z_usamp, "Ref:Acel", "Undersampling in z-direction."),
		OPT_INFILE('C', &custom_angle_file, "file", "custom_angle file [phi + i * psi]"),
		OPT_INFILE('V', &gdelays_file, "file", "(custom_gdelays)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	// Load custom_angle
	long sdims[DIMS];
	complex float* custom_angle_vals = NULL;

	if (NULL != custom_angle_file) {

		if (!conf.radial)
			error("Custom angles make sense only for radial trajectories!");

		debug_printf(DP_INFO, "Custom-angle file is used.\n");

		custom_angle_vals = load_cfl(custom_angle_file, DIMS, sdims);

		if (Y != sdims[0])
			debug_printf(DP_INFO, "According to the custom angle file : number of projection (y) = %d\n", sdims[0]);

		Y = sdims[0];
	}

	if (over <= 0.)
		error("Oversampling factor must be positive.\n");

	X *= over;

	int tot_sp = Y * E * mb * turns;	// total number of lines/spokes
	int N = X * tot_sp / conf.accel;


	long dims[DIMS] = { [0 ... DIMS - 1] = 1  };
	dims[0] = 3;
	dims[1] = X;
	dims[2] = (conf.radial ? Y : (Y / conf.accel));

	dims[TE_DIM] = E;

	if (-1 == D)
		D = X;

	if (D < X)
		error("actual readout samples must be less than full samples");

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


	long gdims[DIMS];
	long gstrs[DIMS];

	complex float* gdelays2 = NULL;

	if (NULL != gdelays_file) {

		// FIXME: check that -q/-Q was not used

		gdelays2 = load_cfl(gdelays_file, DIMS, gdims);

		assert((3 == gdims[0] || (6 == gdims[0])));
		assert(md_check_compat(DIMS - 1, ~0L, dims + 1, gdims + 1));

		md_calc_strides(DIMS, gstrs, gdims, sizeof(complex float));
	}


	complex float* samples = create_cfl(out_file, DIMS, dims);

	md_clear(DIMS, dims, samples, CFL_SIZE);

	double golden_ratio = (sqrtf(5.) + 1.) / 2;
	double angle_atom = M_PI / Y;

	double base_angle[DIMS] = { 0. };
	calc_base_angles(base_angle, Y, E, mb2, turns, conf);

	int p = 0;
	long pos[DIMS] = { 0 };
	double phin1 = 0;

	do {
		int i = pos[PHS1_DIM];
		int j = pos[PHS2_DIM] * conf.accel;
		int e = pos[TE_DIM];
		int m = pos[SLICE_DIM];

		if (conf.radial) {

			/* Calculate read-out samples
			 * for symmetric trajectory [DC between between sample no. X/2-1 and X/2, zero-based indexing]
			 * or asymmetric trajectory [DC component at sample no. X/2, zero-based indexing]
			 */
			int sample = i + D - X;

			if (conf.mems_traj && (1 == e % 2))
			       sample =	D - i;

			double read = (float)sample + (conf.asym_traj ? 0 : 0.5) - (float)D / 2.;


			if (conf.golden_partition)
				base_angle[SLICE_DIM] = (m > 0) ? (fmod(angle_atom * m / golden_ratio, angle_atom) / m) : 0;

			double angle = 0.;

			for (int d = 1; d < DIMS; d++)
				angle += pos[d] * base_angle[d];


			if (conf.half_circle_gold)
				angle = fmod(angle, M_PI);

			angle += M_PI * rot / 180.;

			// 3D
			double angle2 = 0.;

			if (conf.d3d) {

				/* Saff EB., Kuijlaars ABJ.
				 * Distributing many points on a sphere.
				 * The Mathematical Intelligencer 1997;19:11.
				 * DOI:10.1007/BF03024331
				 */

				int Y2 = Y;

				if (!conf.full_circle) // half sphere
					Y2 = Y * 2;

				double hn = -1.0 + (double)(2 * j) / (Y2 - 1);

				if ((j + 1 == Y) || (j == 0))
					angle = 0;
				else
					angle = fmod(phin1 + 3.6 / sqrt(Y2 * (1.0 - hn * hn)), 2. * M_PI);

				if (i + 1 == X)	// FIXME: a non-recursive formula?
					phin1 = angle;

				angle2 = acos(hn) - M_PI / 2.;

				if (NULL != custom_angle_vals)
					angle2 = cimagf(custom_angle_vals[j]);
			}


			if (NULL != custom_angle_vals)
				angle = crealf(custom_angle_vals[j]);


			if (NULL != gdelays2) {

				for (pos[0] = 0; pos[0] < gdims[0]; pos[0]++) {

					assert(pos[0] < 6);
					int a = pos[0] / 3;
					int b = pos[0] % 3;

					gdelays[a][b] = crealf(MD_ACCESS(DIMS, gstrs, pos, gdelays2));
				}

				pos[0] = 0;
			}

			float d[3] = { 0., 0., 0 };
			gradient_delay(d, gdelays, angle, angle2);

			float read_dir[3];
			euler(read_dir, angle, angle2);

			if (!conf.transverse) {

				// project to read direction

				float delay = 0.;

				for (int i = 0; i < 3; i++)
					delay += read_dir[i] * d[i];

				for (int i = 0; i < 3; i++)
					d[i] = delay * read_dir[i];
			}

			samples[p * 3 + 0] = (d[1] + read * read_dir[1]) / over;
			samples[p * 3 + 1] = (d[0] + read * read_dir[0]) / over;
			samples[p * 3 + 2] = (d[2] + read * read_dir[2]) / over;

		} else {

			double x = (i - X / 2) / over;
			double y = (j - Y / 2);
			double angle = -rot / 180. * M_PI;

			samples[p * 3 + 0] =  x * cos(angle) + -y * sin(angle);
			samples[p * 3 + 1] =  x * sin(angle) + y * cos(angle);
			samples[p * 3 + 2] = 0;
		}

		p++;

	} while(md_next(DIMS, dims, ~1L, pos));

	assert(p == N - 0);

	if (NULL != gdelays2)
		unmap_cfl(DIMS, gdims, gdelays2);

	if (NULL != custom_angle_vals)
		unmap_cfl(DIMS, sdims, custom_angle_vals);

	unmap_cfl(DIMS, dims, samples);

	return 0;
}




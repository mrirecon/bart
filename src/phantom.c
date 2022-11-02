/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/rand.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/phantom.h"



static const char help_str[] = "Image and k-space domain phantoms.";



int main_phantom(int argc, char* argv[argc])
{
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool kspace = false;
	bool d3 = false;
	int sens = 0;
	int osens = -1;
	int xdim = -1;

	int geo = -1;

	enum ptype_e { SHEPPLOGAN, CIRC, TIME, SENS, GEOM, STAR, BART, TUBES, RAND_TUBES, NIST, SONAR } ptype = SHEPPLOGAN;

	const char* traj_file = NULL;
	bool basis = false;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 128;
	dims[1] = 128;
	dims[2] = 1;

	int rinit = -1;
	int N = -1;

	float rotation_angle = 0.;
	int rotation_steps = 1;


	const struct opt_s opts[] = {

		OPT_INT('s', &sens, "nc", "nc sensitivities"),
		OPT_INT('S', &osens, "nc", "Output nc sensitivities"),
		OPT_SET('k', &kspace, "k-space"),
		OPT_INFILE('t', &traj_file, "file", "trajectory"),
		OPT_SELECT('c', enum ptype_e, &ptype, CIRC, "()"),
		OPT_SELECT('a', enum ptype_e, &ptype, STAR, "()"),
		OPT_SELECT('m', enum ptype_e, &ptype, TIME, "()"),
		OPT_SELECT('G', enum ptype_e, &ptype, GEOM, "geometric object phantom"),
		OPT_SELECT('T', enum ptype_e, &ptype, TUBES, "tubes phantom"),
		OPTL_SELECT(0, "NIST", enum ptype_e, &ptype, NIST, "NIST phantom (T2 sphere)"),
                OPTL_SELECT(0, "SONAR", enum ptype_e, &ptype, SONAR, "Diagnostic Sonar phantom"),
		OPT_INT('N', &N, "num", "Random tubes phantom and number"),
		OPT_SELECT('B', enum ptype_e, &ptype, BART, "BART logo"),
		OPT_INT('x', &xdim, "n", "dimensions in y and z"),
		OPT_INT('g', &geo, "n=1,2,3", "select geometry for object phantom"),
		OPT_SET('3', &d3, "3D"),
		OPT_SET('b', &basis, "basis functions for geometry"),
		OPT_INT('r', &rinit, "seed", "random seed initialization"),
		OPTL_FLOAT(0, "rotation-angle", &(rotation_angle), "[deg]", "Angle of Rotation"),
		OPTL_INT(0, "rotation-steps", &(rotation_steps), " ", "Number of rotation steps"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (-1 != rinit)
		num_rand_init(rinit);

	if (-1 != N) {

		ptype = RAND_TUBES;
		if (N > 200)
			BART_WARN("Number of tubes is large. Runetime may be very slow.\n");

	} else {

		N = (SONAR == ptype ? 8 : (NIST == ptype ? 15 : (BART == ptype ? 6 : 11)));
	}

	if ((GEOM != ptype) && (-1 != geo)) {

		assert(SHEPPLOGAN == ptype);
		ptype = GEOM;
	}

	if ((GEOM == ptype) && (-1 == geo))
		geo = 1;


	if (TIME == ptype)
		dims[TE_DIM] = 32;

	if ((TUBES == ptype) || (NIST == ptype) || (SONAR == ptype))
		dims[TIME_DIM] = rotation_steps;

	if (-1 != osens) {

		assert(SHEPPLOGAN == ptype);
		ptype = SENS;
		sens = osens;
	}

	if (-1 != xdim)
		dims[0] = dims[1] = xdim;

	if (d3)
		dims[2] = dims[0];


	long sdims[DIMS];
	long sstrs[DIMS] = { 0 };
	complex float* samples = NULL;

	if (NULL != traj_file) {

		if (-1 != xdim)
			debug_printf(DP_WARN, "size ignored.\n");

		kspace = true;

		samples = load_cfl(traj_file, DIMS, sdims);

		md_calc_strides(DIMS, sstrs, sdims, sizeof(complex float));

		dims[0] = 1;
		dims[1] = sdims[1];
		dims[2] = sdims[2];

		dims[TE_DIM] = sdims[TE_DIM];

		dims[TIME_DIM] = sdims[TIME_DIM];
	}

	if (sens > 0)
		dims[3] = sens;

	if (basis) {

		assert(TUBES == ptype || RAND_TUBES == ptype || NIST == ptype || SONAR == ptype || BART == ptype);
		dims[COEFF_DIM] = N; // Number of elements of tubes phantom with rings see src/shepplogan.c
	}


	complex float* out = create_cfl(out_file, DIMS, dims);

	md_clear(DIMS, dims, out, sizeof(complex float));


	switch (ptype) {

	case SENS:

		assert(NULL == traj_file);
		assert(!kspace);

		calc_sens(dims, out);
		break;

	case GEOM:

		if ((geo < 1) || (geo > 3))
			error("geometric phantom: invalid geometry");

		if (d3)
			error("geometric phantom: no 3D mode");

		calc_geo_phantom(dims, out, kspace, geo, sstrs, samples);
		break;

	case STAR:

		assert(!d3);
		calc_star(dims, out, kspace, sstrs, samples);
		break;

	case TIME:

		assert(!d3);
		calc_moving_circ(dims, out, kspace, sstrs, samples);
		break;

	case CIRC:

		calc_circ(dims, out, d3, kspace, sstrs, samples);
//		calc_ring(dims, out, kspace);
		break;

	case SHEPPLOGAN:

		calc_phantom(dims, out, d3, kspace, sstrs, samples);
		break;

	case TUBES:
	case NIST:
        case SONAR:

		calc_phantom_tubes(dims, out, kspace, false, rotation_angle, N, sstrs, samples);
		break;

	case RAND_TUBES:

		calc_phantom_tubes(dims, out, kspace, true, rotation_angle, N, sstrs, samples);
		break;

	case BART:

		calc_bart(dims, out, kspace, sstrs, samples);
		break;
	}

	if (NULL != samples)
		unmap_cfl(3, sdims, samples);

	unmap_cfl(DIMS, dims, out);

	return 0;
}



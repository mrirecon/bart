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

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "simu/phantom.h"



static const char usage_str[] = "<output>";
static const char help_str[] = "Image and k-space domain phantoms.";



int main_phantom(int argc, char* argv[])
{
	bool kspace = false;
	bool d3 = false;
	int sens = 0;
	int osens = -1;
	int xdim = -1;

	int geo = -1;
	enum ptype_e { SHEPPLOGAN, CIRC, TIME, SENS, GEOM, STAR, BART, TUBES } ptype = SHEPPLOGAN;

	const char* traj = NULL;
	bool basis = false;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 128;
	dims[1] = 128;
	dims[2] = 1;


	const struct opt_s opts[] = {

		OPT_INT('s', &sens, "nc", "nc sensitivities"),
		OPT_INT('S', &osens, "nc", "Output nc sensitivities"),
		OPT_SET('k', &kspace, "k-space"),
		OPT_STRING('t', &traj, "file", "trajectory"),
		OPT_SELECT('c', enum ptype_e, &ptype, CIRC, "()"),
		OPT_SELECT('a', enum ptype_e, &ptype, STAR, "()"),
		OPT_SELECT('m', enum ptype_e, &ptype, TIME, "()"),
		OPT_SELECT('G', enum ptype_e, &ptype, GEOM, "geometric object phantom"),
		OPT_SELECT('T', enum ptype_e, &ptype, TUBES, "tubes phantom"),
		OPT_SELECT('B', enum ptype_e, &ptype, BART, "BART logo"),
		OPT_INT('x', &xdim, "n", "dimensions in y and z"),
		OPT_INT('g', &geo, "n=1,2", "select geometry for object phantom"),
		OPT_SET('3', &d3, "3D"),
		OPT_SET('b', &basis, "basis functions for geometry"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if ((GEOM != ptype) && (-1 != geo)) {

		assert(SHEPPLOGAN == ptype);
		ptype = GEOM;
	}

	if ((GEOM == ptype) && (-1 == geo))
		geo = 1;


	if (TIME == ptype)
		dims[TE_DIM] = 32;

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

	if (NULL != traj) {

		if (-1 != xdim)
			debug_printf(DP_WARN, "size ignored.\n");

		kspace = true;

		samples = load_cfl(traj, DIMS, sdims);

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

		assert(TUBES == ptype);

		dims[COEFF_DIM] = 11; // Number of elements of tubes phantom with rings see src/shepplogan.c
	}


	complex float* out = create_cfl(argv[1], DIMS, dims);

	md_clear(DIMS, dims, out, sizeof(complex float));


	switch (ptype) {

	case SENS:

		assert(NULL == traj);
		assert(!kspace);

		calc_sens(dims, out);
		break;

	case GEOM:

		if ((geo < 1) || (geo > 2))
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

		calc_phantom_tubes(dims, out, kspace, sstrs, samples);
		break;

	case BART:

		calc_bart(dims, out, kspace, sstrs, samples);
		break;
	}

	xfree(traj);

	if (NULL != samples)
		unmap_cfl(3, sdims, samples);

	unmap_cfl(DIMS, dims, out);

	return 0;
}



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

	enum ptype_e { SHEPPLOGAN, CIRC, TIME, SENS, GEOM, STAR, BART, BRAIN, TUBES, RAND_TUBES, NIST, SONAR, FILE } ptype = SHEPPLOGAN;

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

	struct pha_opts popts = pha_opts_defaults;

	const char* file_load = NULL;

	struct opt_s coil_opts[] = {

		OPTL_SELECT(0, "HEAD_2D_8CH", enum coil_type, &(popts.stype), HEAD_2D_8CH, "2D head coil with up to 8 channels"),
		OPTL_SELECT(0, "HEAD_3D_64CH", enum coil_type, &(popts.stype), HEAD_3D_64CH, "3D head coil with up to 64 channels"),
	};

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
		OPTL_SELECT(0, "BRAIN", enum ptype_e, &ptype, BRAIN, "BRAIN geometry phantom"),
		OPT_INT('N', &N, "num", "Random tubes phantom and number"),
		OPT_SELECT('B', enum ptype_e, &ptype, BART, "BART logo"),
		OPTL_INFILE(0, "FILE", (const char**)(&(file_load)), "name", "Arbitrary geometry based on multicfl file."),
		OPT_INT('x', &xdim, "n", "dimensions in y and z"),
		OPT_INT('g', &geo, "n=1,2,3", "select geometry for object phantom"),
		OPT_SET('3', &d3, "3D"),
		OPT_SET('b', &basis, "basis functions for geometry"),
		OPT_INT('r', &rinit, "seed", "random seed initialization"),
		OPTL_FLOAT(0, "rotation-angle", &(rotation_angle), "[deg]", "Angle of Rotation"),
		OPTL_INT(0, "rotation-steps", &(rotation_steps), " ", "Number of rotation steps"),
		OPTL_SUBOPT(0, "coil", "...", "configure type of coil", ARRAY_SIZE(coil_opts), coil_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (-1 != rinit)
		num_rand_init(rinit);

	if (-1 != N) {

		ptype = RAND_TUBES;
		if (N > 200)
			BART_WARN("Number of tubes is large. Runtime may be very slow.\n");

	} else {

		N = (SONAR == ptype ? 8 : (NIST == ptype ? 15 : (BART == ptype ? 6 : (BRAIN == ptype ? 4 : 11))));
	}

	// Load multi cfl geometry file, if provided

	int N_max = 2;
	int D_max = 16;
	int D[N_max];

	long hdims[N_max][D_max];
	const long* store_dims[N_max];

	complex float* multifile[N_max];

	int subfiles = 0;

	if (NULL != file_load) {

		ptype = FILE;

		subfiles = load_multi_cfl(file_load, N_max, D_max, D, hdims, multifile);

		if (subfiles != N_max)
			error("Number of cfls in input does not match required number!");

		for (int i = 0; i < subfiles; i++)
			store_dims[i] = hdims[i];

		N = hdims[1][0];
	}

	if ((GEOM != ptype) && (-1 != geo)) {

		assert(SHEPPLOGAN == ptype);
		ptype = GEOM;
	}

	if ((GEOM == ptype) && (-1 == geo))
		geo = 1;


	if (TIME == ptype)
		dims[TIME_DIM] = 32;

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


	if ((DEFAULT == popts.stype) && (0 < sens || 0 < osens)) {

		if (d3)
			popts.stype = HEAD_3D_64CH;
		else
			popts.stype = HEAD_2D_8CH;
	}

	if ((HEAD_2D_8CH == popts.stype) && (8 < sens || 8 < osens))
		error("More than eight 2D sensitivities are not supported!\n");

	if (((HEAD_2D_8CH == popts.stype) && d3) && (0 < sens || 0 < osens))
		debug_printf(DP_WARN, "A 3D simulation with 2D sensitivities is chosen!\n");


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

		// FIXME, check with previos
		dims[TE_DIM] = sdims[TE_DIM];
		dims[TIME_DIM] = sdims[TIME_DIM];
	}

	if (sens > 0)
		dims[3] = sens;

	if (basis) {

		assert(TUBES == ptype || RAND_TUBES == ptype || NIST == ptype || SONAR == ptype || BART == ptype || BRAIN == ptype || FILE == ptype);
		dims[COEFF_DIM] = N; // Number of elements of tubes phantom with rings see src/shepplogan.c
	}


	complex float* out = create_cfl(out_file, DIMS, dims);

	md_clear(DIMS, dims, out, sizeof(complex float));


	switch (ptype) {

	case SENS:

		assert(NULL == traj_file);
		assert(!kspace);

		calc_sens(dims, out, &popts);
		break;

	case GEOM:

		if ((geo < 1) || (geo > 3))
			error("geometric phantom: invalid geometry");

		if (d3)
			error("geometric phantom: no 3D mode");

		calc_geo_phantom(dims, out, kspace, geo, sstrs, samples, &popts);
		break;

	case STAR:

		assert(!d3);
		calc_star(dims, out, kspace, sstrs, samples, &popts);
		break;

	case TIME:

		assert(!d3);
		calc_moving_circ(dims, out, kspace, sstrs, samples, &popts);
		break;

	case CIRC:

		calc_circ(dims, out, d3, kspace, sstrs, samples, &popts);
//		calc_ring(dims, out, kspace);
		break;

	case SHEPPLOGAN:

		calc_phantom(dims, out, d3, kspace, sstrs, samples, &popts);
		break;

	case TUBES:
	case NIST:
        case SONAR:

		calc_phantom_tubes(dims, out, kspace, false, rotation_angle, N, sstrs, samples, &popts);
		break;

	case RAND_TUBES:

		calc_phantom_tubes(dims, out, kspace, true, rotation_angle, N, sstrs, samples, &popts);
		break;

	case BART:

		calc_bart(dims, out, kspace, sstrs, samples, &popts);
		break;

	case BRAIN:

		calc_brain(dims, out, kspace, sstrs, samples, &popts);
		break;

	case FILE:

		calc_cfl_geom(dims, out, kspace, sstrs, samples, N_max, D_max, hdims, multifile, &popts);
		break;
	}

	if (NULL != samples)
		unmap_cfl(3, sdims, samples);

	if (NULL != file_load)
		unmap_multi_cfl(subfiles, D, store_dims, multifile);

	unmap_cfl(DIMS, dims, out);

	return 0;
}



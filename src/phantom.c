/* Copyright 2014-2020. The Regents of the University of California.
 * Copyright 2015-2025. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
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
#include "stl/misc.h"


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

	enum phantom_type ptype = SHEPPLOGAN;

	const char* stl_file = NULL;
	const char* traj_file = NULL;
	bool basis = false;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 128;
	dims[1] = 128;
	dims[2] = 1;

	unsigned long long randseed = 0;
	int N = -1;

	float rotation_angle = 0.;
	int rotation_steps = 1;

	long ellipsoid_center[3] = { -1, -1, -1 };
	float ellipsoid_axes[3] = { 1, 1, 1 };

	struct coil_opts copts = coil_opts_pha_defaults;

	const char* file_load = NULL;

	struct opt_s coil_opts[] = {

		OPTL_SELECT(0, "HEAD_2D_8CH", enum coil_type, &(copts.ctype), HEAD_2D_8CH, "2D head coil with up to 8 channels"),
		OPTL_SELECT(0, "HEAD_3D_64CH", enum coil_type, &(copts.ctype), HEAD_3D_64CH, "3D head coil with up to 64 channels"),
	};

	const struct opt_s opts[] = {

		OPT_PINT('s', &sens, "nc", "nc sensitivities"),
		OPT_PINT('S', &osens, "nc", "Output nc sensitivities"),
		OPT_SET('k', &kspace, "k-space"),
		OPT_INFILE('t', &traj_file, "file", "trajectory"),
		OPT_SELECT('c', enum phantom_type, &ptype, CIRC, "()"),
		OPT_SELECT('a', enum phantom_type, &ptype, STAR, "()"),
		OPT_SELECT('m', enum phantom_type, &ptype, TIME, "()"),
		OPT_SELECT('G', enum phantom_type, &ptype, GEOM, "geometric object phantom"),
		OPT_SELECT('T', enum phantom_type, &ptype, TUBES, "tubes phantom"),
		OPTL_INFILE(0, "stl", &stl_file, "file", "path to stl file"),
		OPTL_SELECT(0, "NIST", enum phantom_type, &ptype, NIST, "NIST phantom (T2 sphere)"),
		OPTL_SELECT(0, "SONAR", enum phantom_type, &ptype, SONAR, "Diagnostic Sonar phantom"),
		OPTL_SELECT(0, "BRAIN", enum phantom_type, &ptype, BRAIN, "BRAIN geometry phantom"),
		OPTL_SELECT(0, "ELLIPSOID", enum phantom_type, &ptype, ELLIPSOID0, "Ellipsoid."),
		OPTL_VEC3(0, "ellipsoid_center", &ellipsoid_center, "", "x,y,z center coordinates of ellipsoid."),
		OPTL_FLVEC3(0, "ellipsoid_axes", &ellipsoid_axes, "", "Axes lengths of ellipsoid."),
		OPT_PINT('N', &N, "num", "Random tubes phantom with num tubes"),
		OPT_SELECT('B', enum phantom_type, &ptype, BART, "BART logo"),
		OPTL_INFILE(0, "FILE", &file_load, "name", "Arbitrary geometry based on multicfl file."),
		OPT_PINT('x', &xdim, "n", "dimensions in y and z"),
		OPT_PINT('g', &geo, "n=1,2,3", "select geometry for object phantom"),
		OPT_SET('3', &d3, "3D"),
		OPT_SET('b', &basis, "basis functions for geometry"),
		OPT_ULLONG('r', &randseed, "", "random seed initialization. '0' uses the default seed."),
		OPTL_FLOAT(0, "rotation-angle", &rotation_angle, "[deg]", "Angle of rotation"),
		OPTL_PINT(0, "rotation-steps", &rotation_steps, "n", "Number of rotation steps"),
		OPTL_SUBOPT(0, "coil", "...", "configure type of coil", ARRAY_SIZE(coil_opts), coil_opts),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();
	num_rand_init(randseed);

	if (NULL != stl_file)
		ptype = STL;

	// avoid model force for stl sampling
	enum coil_type ctype_ = copts.ctype;

	if (-1 != N) {

		ptype = RAND_TUBES;

		if (0 == N)
			error("Number of tubes must be larger than zero.\n");

		if (N > 200)
			debug_printf(DP_WARN, "Number of tubes is large. Runtime may be very slow.\n");
	}

	const int coeff[] = { [SONAR] = 8, [NIST] = 15, [BART] = 6, [BRAIN] = 4 };

	switch (ptype) {

	case SONAR:
	case NIST:
	case BART:
	case BRAIN:

		N = coeff[ptype];
		break;

	case RAND_TUBES:
		// already set above
		break;

	default:
		N = 11;
	}

	// Load multi cfl geometry file, if provided

	enum { D_max = 16 };
	int D_dim[2];

	long hdims[2][D_max];
	const long *store_dims[2] = { hdims[0], hdims[1] };
	complex float* multifile[2];

	if (NULL != file_load) {

		ptype = GEOMFILE;

		int subfiles = load_multi_cfl(file_load, 2, D_max, D_dim, hdims, multifile);

		if (2 != subfiles)
			error("Number of cfls in input does not match required number!\n");

		N = store_dims[1][0];
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
	else if (rotation_steps > 1)
		debug_printf(DP_WARN, "Parameter for rotation steps only supported for NIST, SONAR, and tubes phantoms.\n");

	if (-1 != osens) {

		assert(SHEPPLOGAN == ptype);
		ptype = SENS;
		sens = osens;
	}

	if (-1 != xdim)
		dims[0] = dims[1] = xdim;

	if (d3)
		dims[2] = dims[0];

	if ((COIL_NONE == copts.ctype) && ((0 < sens) || (0 < osens)))
		copts.ctype = d3 ? HEAD_3D_64CH : HEAD_2D_8CH;

	if ((HEAD_2D_8CH == copts.ctype) && ((8 < sens) || (8 < osens)))
		error("More than eight 2D sensitivities are not supported!\n");

	if ((HEAD_2D_8CH == copts.ctype) && d3 && ((0 < sens) || (0 < osens)))
		debug_printf(DP_WARN, "A 3D simulation with 2D sensitivities is chosen!\n");

	if (ELLIPSOID0 == ptype) {

		if (-1 == ellipsoid_center[0]) {

			ellipsoid_center[0] = dims[0] / 2;
			ellipsoid_center[1] = dims[1] / 2;
			ellipsoid_center[2] = dims[2] / 2;
		}
	}

	long sdims[DIMS];
	long sstrs[DIMS] = { };
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

		// FIXME, check with previous
		dims[TE_DIM] = sdims[TE_DIM];
		dims[TIME_DIM] = sdims[TIME_DIM];
	}

	if (sens > 0)
		dims[3] = sens;

	if (basis) {

		assert(TUBES == ptype || RAND_TUBES == ptype || NIST == ptype || SONAR == ptype || BART == ptype || BRAIN == ptype || GEOMFILE == ptype);
		dims[COEFF_DIM] = N; // Number of elements of tubes phantom with rings see src/shepplogan.c
	}


	complex float* out = create_cfl(out_file, DIMS, dims);

	md_clear(DIMS, dims, out, sizeof(complex float));


	switch (ptype) {

	case ELLIPSOID0:

		calc_ellipsoid(DIMS, dims, out, d3, kspace, sdims, sstrs, samples, ellipsoid_axes, ellipsoid_center, rotation_angle, &copts);
		break;

	case SENS:

		assert(NULL == traj_file);
		assert(!kspace);

		calc_sens(dims, out, &copts);
		break;

	case GEOM:

		if ((geo < 1) || (geo > 3))
			error("geometric phantom: invalid geometry\n");

		if (d3)
			error("geometric phantom: no 3D mode\n");

		calc_geo_phantom(dims, out, kspace, geo, sstrs, samples, &copts);
		break;

	case STAR:

		assert(!d3);
		calc_star(dims, out, kspace, sstrs, samples, &copts);
		break;

	case TIME:

		assert(!d3);
		calc_moving_circ(dims, out, kspace, sstrs, samples, &copts);
		break;

	case CIRC:

		calc_circ(dims, out, d3, kspace, sstrs, samples, &copts);
//		calc_ring(dims, out, kspace);
		break;

	case SHEPPLOGAN:

		calc_phantom(dims, out, d3, kspace, sstrs, samples, &copts);
		break;

	case TUBES:
	case NIST:
        case SONAR:

		calc_phantom_tubes(dims, out, kspace, false, rotation_angle, N, sstrs, samples, &copts);
		break;

	case RAND_TUBES:

		calc_phantom_tubes(dims, out, kspace, true, rotation_angle, N, sstrs, samples, &copts);
		break;

	case BART:

		calc_bart(dims, out, kspace, sstrs, samples, &copts);
		break;

	case BRAIN:

		calc_brain(dims, out, kspace, sstrs, samples, &copts);
		break;

	case GEOMFILE:

		calc_cfl_geom(dims, out, kspace, sstrs, samples, D_max, hdims, multifile, &copts);
		break;

	case STL:

		// prepare phantom sampling grid
		struct grid_opts gopts = grid_opts_defaults;
		gopts.kspace = kspace;

		if (NULL == samples) {

			long gd = 0 > xdim ? 128 : xdim;

			gopts.dims[0] = gd;
			gopts.dims[1] = gd;

			if (d3) {

				gopts.dims[2] = gd;
				gopts.b2[2] = 0.5;
			}
		}

		long gdims[DIMS];
		float* grid = compute_grid(DIMS, gdims, &gopts, sdims, samples);

		// prepare sensitivity maps and sensitivity sampling grid
		struct coil_opts coptss = coil_opts_defaults;

		if (0 < sens)
			coptss.flags = MD_BIT(sens) - 1;

		if (0 < sens && COIL_NONE == ctype_)
			ctype_ = HEAD_2D_8CH;

		coptss.ctype = ctype_;
		coptss.kspace = kspace;
		cnstr_coils(DIMS, &coptss, false);

		struct grid_opts cgopts = gopts;
		long stdims[DIMS];
		float* straj = create_senstraj(DIMS, stdims, &cgopts, &coptss);

		// prepare phantom
		struct phantom_opts popts = phantom_opts_defaults;
		popts.kspace = kspace;

		long stldims[DIMS];
		double* model = NULL;

		if (stl_fileextension(stl_file)) {

			model = stl_read(DIMS, stldims, stl_file);

		} else {

			complex float* cmodel = load_cfl(stl_file, DIMS, stldims);
			model = stl_cfl2d(DIMS, stldims, cmodel);
			unmap_cfl(DIMS, stldims, cmodel);
		}

		stl_compute_normals(DIMS, stldims, model);
		phantom_stl_init(&popts, DIMS, stldims, model);

		long odims[DIMS];
		complex double* cdout = sample_signal(DIMS, odims, gdims, grid, stdims, straj, &popts, &gopts, &coptss);

		assert(md_check_equal_dims(DIMS, odims, dims, ~0UL));

		coptss.dstr(&coptss);
		popts.dstr(&popts);

		md_free(model);
		md_free(straj);
		md_free(grid);

		long pos[DIMS], ostrs[DIMS], cdostrs[DIMS];
		md_set_dims(DIMS, pos, 0);
		md_calc_strides(DIMS, ostrs, odims, CFL_SIZE);
		md_calc_strides(DIMS, cdostrs, odims, CDL_SIZE);

		do {
			MD_ACCESS(DIMS, ostrs, pos, out) = MD_ACCESS(DIMS, cdostrs, pos, cdout);

		} while(md_next(DIMS, odims, ~0UL, pos));

		md_free(cdout);

		break;
	}

	if (NULL != samples)
		unmap_cfl(DIMS, sdims, samples);

	if (NULL != file_load)
		unmap_multi_cfl(2, D_dim, store_dims, multifile);

	unmap_cfl(DIMS, dims, out);

	return 0;
}


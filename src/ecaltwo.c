/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2016 Martin Uecker
 */

#include <complex.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/fft.h"
#include "num/init.h"

#include "calib/calib.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif



static const char help_str[] =
		"Second part of ESPIRiT calibration.\n"
		"Optionally outputs the eigenvalue maps.";





int main_ecaltwo(int argc, char* argv[argc])
{
	long x = 0;
	long y = 0;
	long z = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;
	const char* emaps_file = NULL;

	struct arg_s args[] = {

		ARG_LONG(true, &x, "x"),
		ARG_LONG(true, &y, "y"),
		ARG_LONG(true, &z, "z"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "sensitivities"),
		ARG_OUTFILE(false, &emaps_file, "ev-maps"),
	};

	long maps = 2; // channels;
	struct ecalib_conf conf = ecalib_defaults;


	const struct opt_s opts[] = {

		OPT_FLOAT('c', &conf.crop, "crop_value", "Crop the sensitivities if the eigenvalue is smaller than {crop_value}."),
		OPT_LONG('m', &maps, "maps", "Number of maps to compute."),
		OPT_SET('S', &conf.softcrop, "Create maps with smooth transitions (Soft-SENSE)."),
		OPT_CLEAR('O', &conf.orthiter, "()"),
		OPT_SET('g', &conf.usegpu, "()"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	bart_use_gpu = conf.usegpu;
	num_init_gpu_support();

	long in_dims[DIMS];

	complex float* in_data = load_cfl(in_file, DIMS, in_dims);

	int channels = 0;

	while (in_dims[3] != (channels * (channels + 1) / 2))
		channels++;

	debug_printf(DP_INFO, "Channels: %d\n", channels);

	assert(maps <= channels);


	long out_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	long map_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	
	out_dims[0] = x;
	out_dims[1] = y;
	out_dims[2] = z;
	out_dims[3] = channels;
	out_dims[4] = maps;

	assert((out_dims[0] >= in_dims[0]));
	assert((out_dims[1] >= in_dims[1]));
	assert((out_dims[2] >= in_dims[2]));


	for (int i = 0; i < 3; i++)
		map_dims[i] = out_dims[i];

	map_dims[3] = 1;
	map_dims[4] = maps;


	complex float* out_data = create_cfl(out_file, DIMS, out_dims);
	complex float* emaps;

	if (emaps_file)
		emaps = create_cfl(emaps_file, DIMS, map_dims);
	else
		emaps = md_alloc(DIMS, map_dims, CFL_SIZE);

	caltwo(&conf, out_dims, out_data, emaps, in_dims, in_data, NULL, NULL);

	if (conf.intensity) {

		debug_printf(DP_DEBUG1, "Normalize...\n");

		normalizel1(DIMS, COIL_FLAG, out_dims, out_data);
	}

	debug_printf(DP_DEBUG1, "Crop maps... (%.2f)\n", conf.crop);

	crop_sens(out_dims, out_data, conf.softcrop, conf.crop, emaps);

	debug_printf(DP_DEBUG1, "Fix phase...\n");

	fixphase(DIMS, out_dims, COIL_DIM, out_data, out_data);

	debug_printf(DP_DEBUG1, "Done.\n");

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	if (7 == argc)
		unmap_cfl(DIMS, map_dims, emaps);
	else
		md_free(emaps);

	return 0;
}



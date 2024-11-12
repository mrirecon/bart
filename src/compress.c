/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include <complex.h>
#include <strings.h>

#include "num/multind.h"
#include "num/compress.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Compress data using a binary mask (pattern)";


int main_compress(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* mask_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_INFILE(true, &mask_file, "mask"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool decompress = false;
	complex float fill = 0.;

	const struct opt_s opts[] = {

		OPT_SET('d', &decompress, "decompress data"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long idims[DIMS];
	long mdims[DIMS];

	complex float* in = load_cfl(in_file, DIMS, idims);
	complex float* mask = load_cfl(mask_file, DIMS, mdims);

	long* index = md_alloc_sameplace(DIMS, mdims, sizeof(long), mask);

	long max = md_compress_mask_to_index(DIMS, mdims, index, mask);

	if (decompress) {

		long odims[DIMS];
		md_decompress_dims(DIMS, odims, idims, mdims);

		complex float* out = create_cfl(out_file, DIMS, odims);

		md_decompress(DIMS, odims, out, idims, in, mdims, index, &fill, CFL_SIZE);

		unmap_cfl(DIMS, odims, out);
	} else {

		long odims[DIMS];
		md_compress_dims(DIMS, odims, idims, mdims, max);

		complex float* out = create_cfl(out_file, DIMS, odims);

		md_compress(DIMS, odims, out, idims, in, mdims, index, CFL_SIZE);

		unmap_cfl(DIMS, odims, out);
	}


	unmap_cfl(DIMS, mdims, mask);
	unmap_cfl(DIMS, idims, in);
	md_free(index);

	return 0;
}



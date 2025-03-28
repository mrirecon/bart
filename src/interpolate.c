/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>
#include <strings.h>

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "num/init.h"
#include "num/multind.h"

#include "linops/linop.h"

#include "motion/affine.h"
#include "motion/displacement.h"
#include "motion/interpolate.h"

#ifndef DIMS
#define DIMS 16
#endif

enum INTERPOLATION_TYPE { INTP_COORDS, INTP_DISPLACEMENT, INTP_AFFINE };

static const char help_str[] = "Interpolate with coordinates, displacement field, or affine transform.";

int main_interpolate(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* input_file = NULL;
	const char* motion_file = NULL;
	const char* output_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "flags"),
		ARG_INFILE(true, &input_file, "input"),
		ARG_INFILE(false, &motion_file, "coordinates (default) / displacement field / affine transform"),
		ARG_OUTFILE(true, &output_file, "output"),
	};

	bool nearest_neighbour = false;
	bool cubic = false;
	long out_dims[3] = { };

	enum INTERPOLATION_TYPE interp_type = INTP_COORDS;

	const struct opt_s opts[] = {

		OPT_VEC3('x', &out_dims, "x:y:z", "output dimensions for affine interpolation or coordinates"),
		OPT_SELECT('A', enum INTERPOLATION_TYPE, &interp_type, INTP_AFFINE, "use affine transform for interpolation"),
		OPT_SELECT('D', enum INTERPOLATION_TYPE, &interp_type, INTP_DISPLACEMENT, "use displacement field for interpolation"),
		OPT_SET('N', &nearest_neighbour, "use nearest neighbour interpolation"),
		OPT_SET('C', &cubic, "use cubic interpolation")
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	assert(!(cubic && nearest_neighbour));
	int order = cubic ? 3 : nearest_neighbour ? 0 : 1;

	num_init_gpu_support();

	long dims[DIMS];
	long odims[DIMS];
	long mdims[DIMS];

	complex float* src_ptr = load_cfl(input_file, DIMS, dims);
	assert(1 == dims[MOTION_DIM]);

	md_copy_dims(DIMS, odims, dims);
	if ( 0 != md_calc_size(3, out_dims) )
		md_copy_dims(3, odims, out_dims);

	complex float* mot_ptr = NULL;

	if (NULL != motion_file) {

		mot_ptr = load_cfl(motion_file, DIMS, mdims);
	} else {

		md_copy_dims(DIMS, mdims, odims);
		mdims[MOTION_DIM] = bitcount(flags);
		mot_ptr = anon_cfl(NULL, DIMS, mdims);
		md_positions(DIMS, MOTION_DIM, flags, dims, mdims, mot_ptr);
	}

	complex float* out_ptr = create_cfl(output_file, DIMS, odims);

	switch (interp_type) {

		case INTP_COORDS:
		{
			const struct linop_s* lop_interp = linop_interpolate_create(MOTION_DIM, flags, order, DIMS, odims, mdims, mot_ptr, dims);
			linop_forward(lop_interp, DIMS, odims, out_ptr, DIMS, dims, src_ptr);
			linop_free(lop_interp);
		}
		break;

		case INTP_DISPLACEMENT:
		{
			const struct linop_s* lop_interp = linop_interpolate_displacement_create(MOTION_DIM, flags, order, DIMS, dims, mdims, mot_ptr, dims);
			linop_forward(lop_interp, DIMS, odims, out_ptr, DIMS, dims, src_ptr);
			linop_free(lop_interp);
		}
		break;

		case INTP_AFFINE:
		{
			assert(3 == md_nontriv_dims(DIMS, mdims));
			assert((3 == mdims[0]) || (4 == mdims[0]));
			assert(4 == mdims[1]);

			complex float affine[12];
			md_copy_block(2, MD_DIMS(0, 0), MD_DIMS(3, 4), affine, mdims, mot_ptr, sizeof(complex float));

			if ((FFT_FLAGS != flags) || ((~FFT_FLAGS) & md_nontriv_dims(DIMS, dims)))
				error("Affine interpolation only supports the first three dimensions and flags must be set to 7.\nUse bart looping for higher dimensions.\n");

			affine_interpolate(order, affine, odims, out_ptr, dims, src_ptr);
		}
	}

	unmap_cfl(DIMS, mdims, mot_ptr);
	unmap_cfl(DIMS, dims, src_ptr);
	unmap_cfl(DIMS, odims, out_ptr);

	return 0;
}

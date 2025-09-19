/* Copyright 2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author: Nick Scholand
 */

#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/init.h"
#include "num/conv.h"
#include "num/morph.h"

#include "misc/mri.h"

#include "nlops/nlop.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = "Perform morphological operators on binary data with odd mask sizes.";



int main_morphop(int argc, char* argv[argc])
{
	int mask_size = -1;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &mask_size, "mask_size"),
		ARG_INFILE(true, &in_file, "binary input"),
		ARG_OUTFILE(false, &out_file, "binary output"),
	};


	enum morph_type { EROSION, DILATION, OPENING, CLOSING, LABEL } morph_type = EROSION;

	enum mask_type { HLINE, VLINE, CROSS, BLOCK, BALL } mask_type = BLOCK;


	const struct opt_s opts[] = {

		OPT_SELECT('e', enum morph_type, &morph_type, EROSION, "EROSION (default)"),
		OPT_SELECT('d', enum morph_type, &morph_type, DILATION, "DILATION"),
		OPT_SELECT('o', enum morph_type, &morph_type, OPENING, "OPENING"),
		OPT_SELECT('c', enum morph_type, &morph_type, CLOSING, "CLOSING"),
		OPT_SELECT('l', enum morph_type, &morph_type, LABEL, "LABEL"),

		OPT_SELECT('B', enum mask_type, &mask_type, BALL, "use BALL structuring element"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int N = DIMS;

	long dims[N];

	complex float* in = load_cfl(in_file, N, dims);

	complex float* out = create_cfl(out_file, N, dims);

	// Only odd mask size values are supported by the convolution
	assert(1 == mask_size % 2);

	// FIXME: Check if data is binary else raise
	// ...

	long mask_dims[N];
	md_set_dims(N, mask_dims, 1);
	mask_dims[READ_DIM] = mask_size;
	mask_dims[PHS1_DIM] = mask_size;
	mask_dims[PHS2_DIM] = (1 != dims[PHS2_DIM]) ? mask_size : 1;

	complex float* mask = md_alloc(DIMS, mask_dims, CFL_SIZE);
	md_clear(N, mask_dims, mask, CFL_SIZE);

	switch (mask_type) {

	case BALL:
		md_free(mask);
		mask = md_structuring_element_ball(N, mask_dims, mask_size / 2, FFT_FLAGS & md_nontriv_dims(N, dims), NULL);
		break;

	case HLINE:
		error("Mask Type is not implemented yet.\n");
		// mask = {{0, 0, 0},
		// 	{1, 1, 1},
		// 	{0, 0, 0}};
		break;

	case VLINE:
		error("Mask Type is not implemented yet.\n");
		// mask = {{0, 1, 0},
		// 	{0, 1, 0},
		// 	{0, 1, 0}};
		break;

	case CROSS:
		error("Mask Type is not implemented yet.\n");
		// mask = {{0, 1, 0},
		// 	{1, 1, 1},
		// 	{0, 1, 0}};
		break;

	case BLOCK:
		md_zfill(N, mask_dims, mask, 1.);
		break;

	default:
		error("Please choose a correct structural element/mask.\n");
		break;
	}

	switch (morph_type) {

	case EROSION:
		md_erosion(N, mask_dims, mask, dims, out, in, CONV_CYCLIC);
		break;

	case DILATION:
		md_dilation(N, mask_dims, mask, dims, out, in, CONV_CYCLIC);
		break;

	case OPENING:
		md_opening(N, mask_dims, mask, dims, out, in, CONV_CYCLIC);
		break;

	case CLOSING:
		md_closing(N, mask_dims, mask, dims, out, in, CONV_CYCLIC);
		break;

	case LABEL:
		md_label(N, dims, out, in, mask_dims, mask);
		break;

	default:
		error("Please choose a morphological operation.\n");
		break;
	}

	md_free(mask);

	unmap_cfl(N, dims, in);
	unmap_cfl(N, dims, out);
	return 0;
}


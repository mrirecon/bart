/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Output BART file as audio samples.";

// from view:src/draw.c
static double clamp(double a, double b, double x)
{
	return (x < a) ? a : ((x > b) ? b : x);
}

int main_tosnd(int argc, char* argv[argc])
{
	const char* in_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
	};

	const struct opt_s opts[] = {

	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	complex float* data = load_cfl(in_file, DIMS, dims);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	long pos[DIMS] = { };

	do {
		uint8_t sample = 256 * clamp(0.5 + creal(MD_ACCESS(DIMS, pos, strs, data)), 0., 1.);

		fwrite(&sample, sizeof(sample), 1, stdout);

	} while (md_next(DIMS, dims, ~0UL, pos));

	unmap_cfl(DIMS, dims, data);

	return 0;
}


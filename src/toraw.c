/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/utils.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Output BART file as raw samples.";

int main_toraw(int argc, char* argv[argc])
{
	const char* in_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
	};

	unsigned int bitwidth = 8;
	bool norm = false;
	bool complex_out = false;

	const struct opt_s opts[] = {

		OPTL_UINT('b', "bitwidth", &bitwidth, "bitwidth", "Number of bits (default: 8)"),
		OPTL_SET('n', "normalize", &norm, "Normalize input"),
		OPTL_SET('c', "complex", &complex_out, "Complex output"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	complex float* data = load_cfl(in_file, DIMS, dims);

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	complex float* data_n = md_alloc(DIMS, dims, CFL_SIZE);

	md_copy(DIMS, dims, data_n, data, CFL_SIZE);

	if (norm)
		normalize(DIMS, ~0UL, dims, data_n);
		
	long pos[DIMS] = { };

	union {

		uint8_t s8;
		uint16_t s16;
		uint32_t s32;

	} sample[2];


	do {

		float rval = 0.5 * (1. + CLAMP(creal(MD_ACCESS(DIMS, pos, strs, data_n)), -1., 1.));
		float ival = 0.5 * (1. + CLAMP(cimag(MD_ACCESS(DIMS, pos, strs, data_n)), -1., 1.));

		switch (bitwidth) {
		case 8:
			sample[0].s8 = (pow(2, bitwidth) - 1) * rval;
			sample[1].s8 = (pow(2, bitwidth) - 1) * ival;
			break;
		case 16:
			sample[0].s16 = (pow(2, bitwidth) - 1) * rval;
			sample[1].s16 = (pow(2, bitwidth) - 1) * ival;
			break;
		case 32:
			sample[0].s32 = (pow(2, bitwidth) - 1) * rval;
			sample[1].s32 = (pow(2, bitwidth) - 1) * ival;
			break;
		default:
			error("Unsupported bitwidth!\n");
		}

		if (1 != fwrite(&(sample[0]), (bitwidth / 8), 1, stdout))
			error("Error writing real to stdout!\n");

		if (complex_out && (1 != fwrite(&(sample[1]), (bitwidth / 8), 1, stdout)))
			error("Error writing complex to stdout!\n");

	} while (md_next(DIMS, dims, ~0UL, pos));

	unmap_cfl(DIMS, dims, data);
	md_free(data_n);

	return 0;
}


/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2023 Philip Schaten <philip.schaten@tugraz.at>
 */

#include <assert.h>
#include <complex.h>

#include "num/nlmeans.h"
#include "num/init.h"
#include "num/multind.h"

#include "misc/debug.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif
#define DIMS 16


static const char help_str[] = "Non-local means filter";


int main_nlmeans(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	long patch_length = 5;
	long patch_dist = 5;
	float h_factor = 0.04;
	float a_factor = -1.;

	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "flags"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {

		OPTL_LONG('p', "patch_length", &patch_length, "int", "patch length"),
		OPTL_LONG('d', "patch_dist", &patch_dist, "int", "patch distance"),
		OPT_FLOAT('H', &h_factor, "h", "NLMeans h"),
		OPT_FLOAT('a', &a_factor, "a", "NLMeans a (stddev for gaussian euclidean distance)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	a_factor = (0 > a_factor) ? ((patch_length > 1) ? (patch_length - 1.) / 4. : 1.) : a_factor;

	long dims[DIMS];

	complex float* in = load_cfl(in_file, DIMS, dims);
	complex float* out = create_cfl(out_file, DIMS, dims);

	md_znlmeans(DIMS, dims, flags, out, in, patch_length, patch_dist, h_factor, a_factor);

	unmap_cfl(DIMS, dims, out);
	unmap_cfl(DIMS, dims, in);

	return 0;
}


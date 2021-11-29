/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdlib.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] = 	"Combine/Split multiple cfl files to one multi-cfl file.\n"
				"In normal usage, the last argument is the combined multi-cfl,\n"
				"with '-s', the first argument is the multi-cfl that is split up";


int main_multicfl(int argc, char* argv[argc])
{
	bool separate = false;

	const struct opt_s opts[] = {

		OPT_SET('s', &separate, "separate"),
	};

	long count = 0;
	const char** cfl_files = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 1, { OPT_INOUTFILE, sizeof(char*), &cfl_files, "cfl" }),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	const int n_single_cfls = count - 1 ;

	if (!separate) {

		const char* multi_file = cfl_files[n_single_cfls];
		const char** single_files = cfl_files;
		int D[n_single_cfls];
		long dims_load[n_single_cfls][DIMS];
		const long* dims_store[n_single_cfls];
		const complex float* x[n_single_cfls];

		for (int i = 0; i < n_single_cfls; i++) {

			D[i] = DIMS;
			x[i] = load_cfl(single_files[i], D[i], dims_load[i]);
			dims_store[i] = dims_load[i];
		}

		dump_multi_cfl(multi_file, n_single_cfls, D, dims_store, x);

		for (int i = 0; i < n_single_cfls; i++)
			unmap_cfl(D[i], dims_load[i], x[i]);

	} else {

		const char* multi_file = cfl_files[0];
		const char** single_files = cfl_files + 1;
		int D_max = DIMS;
		int D[n_single_cfls];
		long dims_load[n_single_cfls][D_max];
		const long* dims_store[n_single_cfls];
		complex float* x[n_single_cfls];

		int N = load_multi_cfl(multi_file, n_single_cfls, D_max, D, dims_load, x);

		if (N != n_single_cfls)
			error("Number of cfls in input does not match no of outputs!");

		for (int i = 0; i < N; i++) {

			dump_cfl(single_files[i], D[i], dims_load[i], x[i]);
			dims_store[i] = dims_load[i];
		}

		unmap_multi_cfl(N, D, dims_store, x);
	}

	xfree(cfl_files);

	return 0;
}


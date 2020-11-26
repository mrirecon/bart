#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <assert.h>

#include "misc/debug.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"



#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<input1> [... <inputN>] <output> / <input> <output1> [... <outputN>]";
static const char help_str[] = "Combine/Split multiple cfl files to one multi-cfl file.\n";


int main_multicfl(int argc, char* argv[argc])
{
	bool seperate = false;

	const struct opt_s opts[] = {

		OPT_SET('s', &seperate, "sepereate"),
	};

	cmdline(&argc, argv, 2, 12, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (!seperate) {

		int N = argc - 2;
		int D[N];
		long dims_load[N][DIMS];
		const long* dims_store[N];
		const complex float* args[N];

		for (int i = 0; i < N; i++) {

			D[i] = DIMS;
			args[i] = load_cfl(argv[i + 1], D[i], dims_load[i]);
			dims_store[i] = dims_load[i];
		}

		dump_multi_cfl(argv[N + 1], N, D, dims_store, args);

		for (int i = 0; i < N; i++)
			unmap_cfl(D[i], dims_load[i], args[i]);

	} else {

		int N_max = argc - 2;
		int D_max = DIMS;
		int D[N_max];
		long dims_load[N_max][D_max];
		const long* dims_store[N_max];
		complex float* args[N_max];

		int N = load_multi_cfl(argv[1], N_max, D_max, D, dims_load, args);
		if(N != N_max)
			error("Number of cfls in input does not match no of outputs!");

		for (int i = 0; i < N; i++) {

			dump_cfl(argv[i + 2], D[i], dims_load[i], args[i]);
			dims_store[i] = dims_load[i];
		}

		unmap_multi_cfl(N, D, dims_store, args);
	}

	return 0;
}



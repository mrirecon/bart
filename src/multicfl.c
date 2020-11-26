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


static const char help_str[] = "Combine/Split multiple cfl files to one multi-cfl file.\n";


int main_multicfl(int argc, char* argv[argc])
{
	bool seperate = false;

	const struct opt_s opts[] = {

		OPT_SET('s', &seperate, "sepereate"),
	};

	long count = 0;
	const char** single_files = NULL;
	const char* multi_file = NULL;

	struct arg_s args[] = {

		ARG_TUPLE(true, &count, 1, OPT_INOUTFILE, sizeof(char*), &single_files, "single cfl "),
		ARG_INOUTFILE(true, &multi_file, "multi cfl"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (!seperate) {

		int D[count];
		long dims_load[count][DIMS];
		const long* dims_store[count];
		const complex float* args[count];

		for (int i = 0; i < count; i++) {

			D[i] = DIMS;
			args[i] = load_cfl(single_files[i], D[i], dims_load[i]);
			dims_store[i] = dims_load[i];
		}

		dump_multi_cfl(multi_file, count, D, dims_store, args);

		for (int i = 0; i < count; i++)
			unmap_cfl(D[i], dims_load[i], args[i]);

	} else {

		int D_max = DIMS;
		int D[count];
		long dims_load[count][D_max];
		const long* dims_store[count];
		complex float* args[count];

		int N = load_multi_cfl(multi_file, count, D_max, D, dims_load, args);
		if(N != count)
			error("Number of cfls in input does not match no of outputs!");

		for (int i = 0; i < N; i++) {

			dump_cfl(single_files[i], D[i], dims_load[i], args[i]);
			dims_store[i] = dims_load[i];
		}

		unmap_multi_cfl(N, D, dims_store, args);
	}

	return 0;
}



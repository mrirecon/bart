/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>
#include <string.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/io.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] =
	"Join input files along {dimensions}. All other dimensions must have the same size.\n"
	"\t Example 1: join 0 slice_001 slice_002 slice_003 full_data\n"
	"\t Example 2: join 0 `seq -f \"slice_%%03g\" 0 255` full_data";



int main_join(int argc, char* argv[argc])
{
	long count = 0;
	int dim = -1;
	const char** in_files = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INT(true, &dim, "dimension"),
		ARG_TUPLE(true, &count, 1, { OPT_INFILE, sizeof(char*), &in_files, "input" }),
		ARG_INOUTFILE(true, &out_file, "output"),
	};


	bool append = false;

	const struct opt_s opts[] = {

		OPT_SET('a', &append, "append - only works for cfl files!"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int N = DIMS;

	assert(dim < N);

	if (append) {

		count += 1;

		assert(count > 1);

		int len = strlen(out_file);
		char buf[len + 5];
		strcpy(buf, out_file);
		strcat(buf, ".cfl");

		if (-1 == access(buf, F_OK)) {

			// make sure we do not have any other file format

			strcpy(buf, out_file);
			strcat(buf, ".coo");
			assert(-1 == access(buf, F_OK));

			strcpy(buf, out_file);
			strcat(buf, ".ra");
			assert(-1 == access(buf, F_OK));

			count--;
			append = false;
		}
	}

	long in_dims[count][N];
	const complex float* in_data[count];

	long offsets[count];
	long sum = 0;

	// figure out size of output
	for (int l = 0, i = 0; i < count; i++) {

		const char* name = NULL;

		if (append && (i == 0)) {

			name = out_file;

		} else {

			name = in_files[l++];
		}

		debug_printf(DP_DEBUG1, "loading %s\n", name);

		in_data[i] = load_cfl(name, N, in_dims[i]);

		offsets[i] = sum;

		sum += in_dims[i][dim];

		for (int j = 0; j < N; j++)
			assert((dim == j) || (in_dims[0][j] == in_dims[i][j]));

		if (append && (i == 0))
			unmap_cfl(N, in_dims[i], in_data[i]);
	}

	long out_dims[N];

	for (int i = 0; i < N; i++)
		out_dims[i] = in_dims[0][i];

	out_dims[dim] = sum;

	if (append) {

		// Here, we need to trick the IO subsystem into absolutely NOT
		// unlinking our input, as the same file is also an output here.
		io_close(out_file);
	}

	complex float* out_data = create_cfl(out_file, N, out_dims);

	long ostr[N];
	md_calc_strides(N, ostr, out_dims, CFL_SIZE);

#pragma omp parallel for
	for (int i = 0; i < count; i++) {

		if (!(append && (0 == i))) {

			long pos[N];
			md_singleton_strides(N, pos);
			pos[dim] = offsets[i];

			long istr[N];
			md_calc_strides(N, istr, in_dims[i], CFL_SIZE);

			md_copy_block(N, pos, out_dims, out_data, in_dims[i], in_data[i], CFL_SIZE);

			unmap_cfl(N, in_dims[i], in_data[i]);

			debug_printf(DP_DEBUG1, "done copying file %d\n", i);
		}
	}

	unmap_cfl(N, out_dims, out_data);

	xfree(in_files);

	return 0;
}



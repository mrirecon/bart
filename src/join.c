/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */


#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <complex.h>
#include <string.h>
#include <getopt.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/debug.h"


#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s dimension <input1> ... <inputn> <output>\n", name);
	fprintf(fp, "\t Example 1: %s 0 slice_001 slice_002 slice_003 full_data\n\n", name);
	fprintf(fp, "\t Example 2: %s 0 `seq -f \"slice_%%03g\" 0 255` full_data\n\n", name);
}

static void help(void)
{
	printf(	"\nJoin input files along {dimensions}. All other dimensions must have the same size.\n");
}


int main_join(int argc, char* argv[])
{
	int c;

	while (-1 != (c = getopt(argc, argv, "h"))) {

		switch (c) {

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind < 3) {
		usage(argv[0], stderr);
		exit(1);
	}

	int N = DIMS;

	int dim = atoi(argv[optind + 0]);
	assert(dim < N);

	int count = argc - optind - 2;

	long in_dims[count][N];
	long offsets[count];
	complex float* idata[count];
	long sum = 0;

	// figure out size of output
	for (int i = 0; i < count; i++) {

		debug_printf(DP_DEBUG1, "loading %s\n", argv[optind + 1 + i]);
		idata[i] = load_cfl(argv[optind + 1 + i], N, in_dims[i]);
		offsets[i] = sum;

		sum += in_dims[i][dim];

		for (int j = 0; j < N; j++)
			assert((dim == j) || (in_dims[0][j] == in_dims[i][j]));
	}

	long out_dims[N];

	for (int i = 0; i < N; i++)
		out_dims[i] = in_dims[0][i];

	out_dims[dim] = sum;

	complex float* out_data = create_cfl(argv[argc - 1], N, out_dims);

	long ostr[N];
	md_calc_strides(N, ostr, out_dims, CFL_SIZE);

#pragma omp parallel for
	for (int i = 0; i < count; i++) {

		long pos[N];
		md_singleton_strides(N, pos);
		pos[dim] = offsets[i];

		long istr[N];
		md_calc_strides(N, istr, in_dims[i], CFL_SIZE);

		md_copy_block(N, pos, out_dims, out_data, in_dims[i], idata[i], CFL_SIZE);
		unmap_cfl(N, in_dims[i], idata[i]);
		debug_printf(DP_DEBUG1, "done copying file %d\n", i);
	}

	unmap_cfl(N, out_dims, out_data);

	exit(0);
}



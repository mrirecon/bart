/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */


#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <complex.h>
#include <getopt.h>

#include "num/multind.h"

#include "misc/mmio.h"


#ifndef DIMS
#define DIMS 16
#endif

static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s dimension <input1> ... <inputn> <output>\n", name);
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
	int count = argc - optind - 2;

	int dim = atoi(argv[optind]);
	assert(dim < N);

        long in_dims[count][N];
	complex float* idata[count];
	long sum = 0;

	for (int i = 0; i < count; i++) {

		idata[i] = load_cfl(argv[optind + 1 + i], N, in_dims[i]);
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
	md_calc_strides(N, ostr, out_dims, sizeof(complex float));
	long opos = 0;

	for (int i = 0; i < count; i++) {

		long istr[N];
		md_calc_strides(N, istr, in_dims[i], sizeof(complex float));

		md_copy2(N, in_dims[i], ostr, (char*)out_data + opos * ostr[dim], istr, idata[i], sizeof(complex float));
		unmap_cfl(N, in_dims[i], idata[i]);

		opos += in_dims[i][dim];
	}

	unmap_cfl(N, out_dims, out_data);

	exit(0);
}



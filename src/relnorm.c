/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <libgen.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "misc/mmio.h"
#include "misc/debug.h"


#ifndef DIMS
#define DIMS 16
#endif


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-h] <input1> <input2>\n", name);
}


int main_relnorm(int argc, char* argv[])
{

	if (0 == strcmp(basename(argv[0]), "relnorm"))
		debug_printf(DP_WARN, "The \'relnorm\' command is deprecated. Use \'nrmse\' instead.\n");

	int c;
	while (-1 != (c = getopt(argc, argv, "h"))) {

		switch (c) {

		case 'h':
			usage(argv[0], stdout);
			printf(  "Compute relative l2 norm of two images: norm(x1 - x2) / norm(x2) \n"
				"\t-h\thelp\n"		);
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 2) {
		usage(argv[0], stderr);
		exit(1);
	}


	int N = DIMS;
	long in1_dims[N];
	long in2_dims[N];

	complex float* in1_data = load_cfl(argv[optind + 0], N, in1_dims);
	complex float* in2_data = load_cfl(argv[optind + 1], N, in2_dims);


	for (int i = 0; i < N; i++)
		assert(in1_dims[i] == in2_dims[i]);

	float value = md_zrnorme(DIMS, in1_dims, in1_data, in2_data);
	printf("%+e\n", value);

	unmap_cfl(N, in1_dims, in1_data);
	unmap_cfl(N, in2_dims, in2_data);

	exit(0);
}



/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <unistd.h>
#include <math.h>

#include "num/multind.h"
#include "num/rand.h"

#include "misc/mmio.h"

#ifndef DIMS
#define DIMS 16
#endif

static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-n var] [-r] <input> <output>\n", name);
}

static void help(const char* name, FILE *fd)
{
	usage(name, fd);
	fprintf(fd, "\nAdd noise with selected variance to input.\n"
		"\t-n\tvariance\tDEFAULT: 1.0\n"
		"\t-r\treal-valued input\n"
		"\t-h\thelp\n");
}

int main(int argc, char* argv[])
{

	int com;
	float var = 1.;
	float spike = 1.;
	bool rvc = false;

	while (-1 != (com = getopt(argc, argv, "hrn:s:S:"))) {

		switch (com) {

		case 's':
			num_rand_init(atoi(optarg));
			break;

		case 'S':
			spike = atof(optarg);
			break;

		case 'r':
			rvc = true;
			break;

		case 'n':
			var = (float)atof(optarg);	
			break;

		case 'h':
			help(argv[0], stdout);
			exit(0);

		default:
			help(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 2) {

//		fprintf(stderr, "Input arguments do not match expected format.\n");
		usage(argv[0], stderr);
		exit(1);
	}


	unsigned int N = DIMS;
	long dims[N];

	complex float* y = load_cfl(argv[optind + 0], N, dims);

	complex float* x = create_cfl(argv[optind + 1], N, dims);

	long T = md_calc_size(N, dims);

	// scale var for complex data
	if (!rvc)
		var = var / 2.f;

	float stdev = sqrtf(var);

	for (long i = 0; i < T; i++) {

		x[i] = y[i];

		if (spike >= uniform_rand())
			x[i] += stdev * gaussian_rand();

		if (rvc)
			x[i] = crealf(x[i]);
	}

	unmap_cfl(N, dims, y);
	unmap_cfl(N, dims, x);
	exit(0);
}





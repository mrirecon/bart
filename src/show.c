/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include <complex.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"


#ifndef DIMS
#define DIMS 16
#endif


static void usage(const char* name, FILE* fd)
{	// FIXME
	fprintf(fd, "Usage: %s <input>\n", name);
}


static void print_cfl(unsigned int N, const long dims[N], const complex float* data)
{
	// find first non-trivial dimension
	unsigned int l = 0;
	while ((l < N - 1) && (1 == dims[l]))
		l++;

	long T = md_calc_size(N, dims);

	for (long i = 0; i < T; i++) {

		printf("%+e%+ei", crealf(data[i]), cimagf(data[i]));
		printf((0 == (i + 1) % dims[l]) ? "\n" : "\t");
	}
}

int main_show(int argc, char* argv[])
{
	int c;
	_Bool meta = false;
	int showdim = -1;

	while (-1 != (c = getopt(argc, argv, "hmd:"))) {

		switch (c) {

		case 'm':
			meta = true;
			break;

		case 'd':
			showdim = atoi(optarg);
			break;

		case 'h':
			usage(argv[0], stdout);
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 1) {

		usage(argv[0], stderr);
		exit(1);
	}


	unsigned int N = DIMS;

	long dims[N];
	complex float* data = load_cfl(argv[optind + 0], N, dims);

	if (-1 != showdim) {

		assert((showdim >= 0) && (showdim < (int)N));
		printf("%ld\n", dims[showdim]);
		goto out;
	}

	if (meta) {

		printf("Type: complex float\n");
		printf("Dimensions: %d\n", N);
		printf("AoD:");

		for (unsigned int i = 0; i < N; i++)
			printf("\t%ld", dims[i]);

		printf("\n");

		goto out;
	}

	print_cfl(N, dims, data);
out:
	unmap_cfl(N, dims, data);
	exit(0);
}



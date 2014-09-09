/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <getopt.h>

#include "num/flpmath.h"

#include "misc/mmio.h"

#define DIMS 16



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-j bitmask] lambda <input> <output>\n", name);
}



int main(int argc, char* argv[])
{
	unsigned int flags = 0;

	char c;
	while (-1 != (c = getopt(argc, argv, "j:h"))) {

		switch (c) {

		case 'j':
			flags = atoi(optarg);
			break;

		case 'h':
			usage(argv[0], stdout);
			printf(	"\nPerform softthresholding with parameter lambda.\n\n"
				"-j bitmask\tjoint thresholding\n"
				"-h\thelp\n"		);
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 3) {

		usage(argv[0], stderr);
		exit(1);
	}

	const int N = DIMS;
	long dims[N];
	complex float* idata = load_cfl(argv[optind + 1], N, dims);
	complex float* odata = create_cfl(argv[optind + 2], N, dims);

	float lambda = atof(argv[optind + 0]);

	md_zsoftthresh(N, dims, lambda, flags, odata, idata);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);
	exit(0);
}



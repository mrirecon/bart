/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/wavelet.h"
#include "misc/mmio.h"
#include "misc/io.h"

#ifndef DIMS
#define DIMS 16
#endif


static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s [-i] bitmask <input> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Perform a wavelet (cdf97) transform.\n"
		"\n"
		"-i\tinverse\n"
		"-h\thelp\n");
}



int main_cdf97(int argc, char* argv[])
{
	int c;
	_Bool inv = false;

	while (-1 != (c = getopt(argc, argv, "ih"))) {

		switch (c) {

		case 'i':
			inv = true;
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
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

	unsigned int flags = atoi(argv[optind + 0]);

	long dims[DIMS];
	complex float* idata = load_cfl(argv[optind + 1], DIMS, dims);
	complex float* odata = create_cfl(argv[optind + 2], DIMS, dims);

	md_copy(DIMS, dims, odata, idata, CFL_SIZE);
	unmap_cfl(DIMS, dims, idata);

	if (inv) {

		md_iresortz(DIMS, dims, flags, odata);
		md_icdf97z(DIMS, dims, flags, odata);

	} else {

		md_cdf97z(DIMS, dims, flags, odata);
		md_resortz(DIMS, dims, flags, odata);
	}

	unmap_cfl(DIMS, dims, odata);
	exit(0);
}






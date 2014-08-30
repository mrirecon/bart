/* Copyright 2013. The Regents of the University of California.
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
#include <stdio.h>
#include <complex.h>
#include <unistd.h>


#include "num/multind.h"
#include "num/fft.h"

#include "misc/mmio.h"

#ifndef DIMS
#define DIMS 16
#endif


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-u] [-i] bitmask <input> <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Performs a fast Fourier transform (FFT) along selected dimensions.\n"
		"\n"
		"-u\tunitary\n"
		"-i\tinverse\n"
		"-h\thelp\n");
}



int main(int argc, char* argv[])
{
	int c;
	bool unitary = false;
	bool inv = false;

	while (-1 != (c = getopt(argc, argv, "uih"))) {

		switch (c) {

		case 'u':
			unitary = true;	
			break;

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



	long dims[DIMS];
	complex float* idata = load_cfl(argv[optind + 1], DIMS, dims);
	complex float* data = create_cfl(argv[optind + 2], DIMS, dims);

	unsigned long flags = labs(atol(argv[optind + 0]));



	md_copy(DIMS, dims, data, idata, sizeof(complex float));
	unmap_cfl(DIMS, dims, idata);

	if (unitary)
		fftscale(DIMS, dims, flags, data, data);

	(inv ? ifftc : fftc)(DIMS, dims, flags, data, data);

	unmap_cfl(DIMS, dims, data);
	exit(0);
}



/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Author; 
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include "num/multind.h"

#include "misc/resize.h"
#include "misc/mmio.h"


#ifndef DIMS
#define DIMS 16
#endif

static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s [-c] dimension size <input> <output>\n", name);
}

static void help(void)
{
	printf(	"\n"
		"Resizes an array along dimension to size by truncating or zero-padding.\n"
		"\n"
		"-c\tcenter\n");
}


int main_resize(int argc, char* argv[])
{
	bool center = false;

	int c;
	while (-1 != (c = getopt(argc, argv, "csh"))) {

		switch (c) {

		case 'c':
			center = true;
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

	if (argc - optind != 4) {

		usage(argv[0], stderr);
		exit(1);
	}

	unsigned int N = DIMS;

	unsigned int dim = atoi(argv[optind + 0]);
	unsigned int count = atoi(argv[optind + 1]);

	assert(dim < N);
	assert(count >= 1);

	long in_dims[N];
	long out_dims[N];

	void* in_data = load_cfl(argv[optind + 2], N, in_dims);
	
	for (unsigned int i = 0; i < N; i++)
		out_dims[i] = in_dims[i];

	out_dims[dim] = count;

	void* out_data = create_cfl(argv[optind + 3], N, out_dims);

	(center ? md_resizec : md_resize)(N, out_dims, out_data, in_dims, in_data, sizeof(complex float));

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);

	exit(0);
}



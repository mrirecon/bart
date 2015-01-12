/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
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

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s [-c] dim1 size1 ... dimn sizen <input> <output>\n", name);
}

static void help(void)
{
	printf(	"\n"
		"Resizes an array along dimensions to sizes by truncating or zero-padding.\n"
		"\n"
		"-c\tcenter\n");
}


int main_resize(int argc, char* argv[])
{
	bool center = false;

	int c;
	while (-1 != (c = getopt(argc, argv, "ch"))) {

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

	if (argc - optind < 4) {

		usage(argv[0], stderr);
		exit(1);
	}

	unsigned int N = DIMS;

	int count = argc - optind - 2;
	assert((count > 0) && (count % 2 == 0));

	long in_dims[N];
	long out_dims[N];

	void* in_data = load_cfl(argv[argc - 2], N, in_dims);
	md_copy_dims(N, out_dims, in_dims);
	
	for (int i = 0; i < count; i += 2) {

		unsigned int dim = atoi(argv[optind + i]);
		unsigned int size = atoi(argv[optind + i + 1]);

		assert(dim < N);
		assert(size >= 1);

		out_dims[dim] = size;
	}

	void* out_data = create_cfl(argv[argc - 1], N, out_dims);

	(center ? md_resizec : md_resize)(N, out_dims, out_data, in_dims, in_data, CFL_SIZE);

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);

	exit(0);
}



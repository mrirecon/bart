/* Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <unistd.h>

#include <complex.h>

#include "num/multind.h"
#include "num/casorati.h"
#include "num/filter.h"

#include "misc/mmio.h"
#include "misc/misc.h"



#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-m] [-l] <input> <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Apply filter.\n"
		"\n"
		"-m d\tmedian filter along dimension d\n"
		"-l l\tlength of filter\n"
		"-h\thelp\n");
}

int main_filter(int argc, char* argv[])
{
	int c;
	int len = -1;
	int dim = -1;

	while (-1 != (c = getopt(argc, argv, "m:l:h"))) {

		switch (c) {

		case 'l':
			len = atoi(optarg);	
			break;

		case 'm':
			dim = atoi(optarg);
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

	if ((0 > dim) || (0 > len) || (argc - optind != 2)) {

		usage(argv[0], stderr);
		exit(1);
	}


	long in_dims[DIMS];
	
	complex float* in_data = load_cfl(argv[optind + 0], DIMS, in_dims);

	assert(dim >= 0);
	assert(dim < DIMS);
	assert(len > 0);
	assert(len <= in_dims[dim]);

	long tmp_dims[DIMS + 1];
	md_copy_dims(DIMS, tmp_dims, in_dims);
	tmp_dims[DIMS] = 1;

	long tmp2_strs[DIMS + 1];
	md_calc_strides(DIMS + 1, tmp2_strs, tmp_dims, CFL_SIZE);

	long tmp_strs[DIMS + 1];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);

	tmp_dims[DIMS] = len;
	tmp_dims[dim] = in_dims[dim] - len + 1;
	tmp_strs[DIMS] = tmp_strs[dim];

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, tmp_dims);

	complex float* out_data = create_cfl(argv[optind + 1], DIMS, out_dims);

	md_medianz2(DIMS + 1, DIMS, tmp_dims, tmp2_strs, out_data, tmp_strs, in_data);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	exit(0);
}



/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <getopt.h>

#include <complex.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"


#define DIMS 16

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char* usage_str = "flags dim1 ... dimN <input> <output>";
static const char* help_str = "Reshape selected dimensions.\n";

static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s %s\n", name, usage_str);
}

static void help(void)
{
	printf("\n%s\n\n", help_str);
}



int main_reshape(int argc, char* argv[])
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

	unsigned int flags = atoi(argv[optind + 0]);
	unsigned int n = bitcount(flags);

	assert((int)n + 3 == argc - optind);

	long in_dims[DIMS];
	long in_strs[DIMS];

	long out_dims[DIMS];
	long out_strs[DIMS];

	complex float* in_data = load_cfl(argv[optind + n + 1], DIMS, in_dims);

	md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

	md_copy_dims(DIMS, out_dims, in_dims);
	
	unsigned int j = 0;

	for (unsigned int i = 0; i < DIMS; i++)
		if (MD_IS_SET(flags, i))
			out_dims[i] = atoi(argv[optind + j++ + 1]);

	assert(j == n);
	assert(md_calc_size(DIMS, in_dims) == md_calc_size(DIMS, out_dims));

	md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);
	
	for (unsigned int i = 0; i < DIMS; i++)
		if (!(MD_IS_SET(flags, i) || (in_strs[i] == out_strs[i]))) 
			error("Dimensions are not consistent at index %d.\n");


	complex float* out_data = create_cfl(argv[optind + n + 2], DIMS, out_dims);

	md_copy(DIMS, in_dims, out_data, in_data, CFL_SIZE);

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	exit(0);
}



/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <complex.h>
#include <stdio.h>
#include <getopt.h>

#include "num/multind.h"
#include "num/conv.h"

#include "misc/mmio.h"

#ifndef DIMS
#define DIMS 16
#endif


static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s bitmask <input> <kernel> <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Performs a convolution along selected dimensions.\n"
		"\n"
		"-h\thelp\n");
}



int main_conv(int argc, char* argv[])
{
	int c;

	while (-1 != (c = getopt(argc, argv, "ah"))) {

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

        if (argc - optind != 4) {

                usage(argv[0], stderr);
                exit(1);
        }

	unsigned int flags = atoi(argv[optind + 0]);

	unsigned int N = DIMS;
	long dims[N];
	const complex float* in = load_cfl(argv[optind + 1], N, dims);

	long krn_dims[N];
	const complex float* krn = load_cfl(argv[optind + 2], N, krn_dims);
	complex float* out = create_cfl(argv[optind + 3], N, dims);

	struct conv_plan* plan = conv_plan(N, flags, CONV_CYCLIC, CONV_SYMMETRIC, dims, dims, krn_dims, krn);
	conv_exec(plan, out, in);
	conv_free(plan);

	unmap_cfl(N, dims, out);
	unmap_cfl(N, krn_dims, krn);
	unmap_cfl(N, dims, in);
	exit(0);
}





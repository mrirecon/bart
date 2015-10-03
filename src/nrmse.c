/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Dara Bahri <dbahri123@gmail.com>
 * 2014 Frank Ong <frankong@berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2015 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>

#include "misc/mmio.h"
#include "misc/misc.h"

#include "num/flpmath.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char* usage_str = "[-h] <reference> <input>";
static const char* help_str = 
	"Output normalized root mean square error (NRMSE),\n"
	"i.e. norm(input - ref) / norm(ref) \n\n"
	"-t\ttest\n"
	"-h\thelp\n";
			
static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s %s\n", name, usage_str);
}

static void help(void)
{
	printf("\n%s", help_str);
}

int main_nrmse(int argc, char* argv[])
{
	int c;
	float test = -1.;

	while (-1 != (c = getopt(argc, argv, "t:h"))) {

		switch (c) {

		case 't':
			test = atof(optarg);
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

	if (argc - optind != 2) {

		usage(argv[0], stderr);
		exit(1);
	}

	long ref_dims[DIMS];
	long in_dims[DIMS];
	complex float* ref = load_cfl(argv[optind + 0], DIMS, ref_dims);
	complex float* in = load_cfl(argv[optind + 1], DIMS, in_dims);

	for (int i = 0; i < DIMS; i++)
		assert(in_dims[i] == ref_dims[i]);

	float err = md_znrmse(DIMS, ref_dims, ref, in);
	printf("%f\n", err);

	unmap_cfl(DIMS, ref_dims, ref);
	unmap_cfl(DIMS, in_dims, in);

	exit(((test == -1.) || (err <= test)) ? 0 : 1);
}




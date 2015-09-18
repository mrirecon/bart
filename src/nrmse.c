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

#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <stdbool.h>
#include <assert.h>

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
	"-h\thelp\n";
			

int main_nrmse(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	long ref_dims[DIMS];
	long in_dims[DIMS];
	complex float* ref = load_cfl(argv[1], DIMS, ref_dims);
	complex float* in = load_cfl(argv[2], DIMS, in_dims);

	for (int i = 0; i < DIMS; i++)
		assert(in_dims[i] == ref_dims[i]);

	printf("%f\n", md_znrmse(DIMS, ref_dims, ref, in));

	unmap_cfl(DIMS, ref_dims, ref);
	unmap_cfl(DIMS, in_dims, in);
	exit(0);
}




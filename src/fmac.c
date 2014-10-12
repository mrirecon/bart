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
#include <complex.h>
#include <stdio.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mmio.h"

#ifndef DIMS
#define DIMS 16
#endif

static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s [-A] [-C] [-s bitmask] <input1> <input2> <output>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Multiply and accumulate.\n"
		"\n"
		"-A\tadd to existing output (instead of overwriting)\n"
		"-C\tconjugate input2\n"
		"-s\tsquash selected dimensions\n");
}


int main_fmac(int argc, char* argv[])
{
	bool clear = true;
	bool conj = false;
	unsigned int squash = 0;

	int c;

	while (-1 != (c = getopt(argc, argv, "hACs:"))) {

		switch (c) {

		case 'A':
			clear = false;
			break;

		case 'C':
			conj = true;
			break;
	
		case 's':
			squash = atoi(optarg);
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

	if (3 != argc - optind) {

		usage(argv[0], stderr);
		exit(1);
	}


	int N = DIMS;

	long dims1[N];
	long dims2[N];

	complex float* data1 = load_cfl(argv[optind + 0], N, dims1);
	complex float* data2 = load_cfl(argv[optind + 1], N, dims2);

	long dims[N];

	for (int i = 0; i < N; i++) {
		
		assert((dims1[i] == dims2[i]) || (1 == dims1[i]) || (1 == dims2[i]));
	
		dims[i] = (1 == dims1[i]) ? dims2[i] : dims1[i];
	}

	long dimso[N];
	md_select_dims(N, ~squash, dimso, dims);
	complex float* out = create_cfl(argv[optind + 2], N, dimso);

	if (clear) {

		md_clear(N, dimso, out, sizeof(complex float));
	}

	long str1[N];
	long str2[N];
	long stro[N];

	md_calc_strides(N, str1, dims1, sizeof(complex float));
	md_calc_strides(N, str2, dims2, sizeof(complex float));
	md_calc_strides(N, stro, dimso, sizeof(complex float));

	(conj ? md_zfmacc2 : md_zfmac2)(N, dims, stro, out, str1, data1, str2, data2);

	unmap_cfl(N, dims1, data1);
	unmap_cfl(N, dims2, data2);
	unmap_cfl(N, dimso, out);
	exit(0);
}



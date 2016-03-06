/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2013, 2015-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<input>";
static const char help_str[] = "Outputs values or meta data.";

static void print_cfl(unsigned int N, const long dims[N], const complex float* data, const char* sep)
{
	// find first non-trivial dimension
	unsigned int l = 0;
	while ((l < N - 1) && (1 == dims[l]))
		l++;

	long T = md_calc_size(N, dims);

	for (long i = 0; i < T; i++) {

		printf("%+e%+ei", crealf(data[i]), cimagf(data[i]));
		printf((0 == (i + 1) % dims[l]) ? "\n" : sep);
	}
}

int main_show(int argc, char* argv[])
{
	bool meta = false;
	int showdim = -1;
	const char* sep = NULL;

	const struct opt_s opts[] = {

		OPT_SET('m', &meta, "show meta data"),
		OPT_INT('d', &showdim, "dim", "show size of dimension"),
		OPT_STRING('s', &sep, "sep", "use <sep> as the separator"),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);


	unsigned int N = DIMS;

	long dims[N];
	complex float* data = load_cfl(argv[1], N, dims);

	if (-1 != showdim) {

		assert((showdim >= 0) && (showdim < (int)N));
		printf("%ld\n", dims[showdim]);
		goto out;
	}

	if (meta) {

		printf("Type: complex float\n");
		printf("Dimensions: %d\n", N);
		printf("AoD:");

		for (unsigned int i = 0; i < N; i++)
			printf("\t%ld", dims[i]);

		printf("\n");

		goto out;
	}

	print_cfl(N, dims, data, NULL == sep ? "\t" : sep);
out:
	unmap_cfl(N, dims, data);
	if (NULL != sep)
		free((void*)sep);

	exit(0);
}



/* Copyright 2013-2016. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2013, 2015-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2015-2016 Jon Tamir <jtamir.eecs.berkeley.edu>
 */

#define _GNU_SOURCE

#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "<input>";
static const char help_str[] = "Outputs values or meta data.";


static void print_cfl(unsigned int N, const long dims[N], const complex float* data, const char* fmt, const char* sep)
{
	// find first non-trivial dimension
	unsigned int l = 0;
	while ((l < N - 1) && (1 == dims[l]))
		l++;

	long T = md_calc_size(N, dims);

	const char* allowed_fmts[] = {

		"%+e%+ei",
		"%+f%+fi",
	};

	for (unsigned int i = 0; i < ARRAY_SIZE(allowed_fmts); i++)
		if (0 == strcmp(allowed_fmts[i], fmt))
			goto ok;

	debug_printf(DP_ERROR, "Invalid format string.\n");
	return;
ok:

	for (long i = 0; i < T; i++) {

		printf(fmt, crealf(data[i]), cimagf(data[i]));
		printf((0 == (i + 1) % dims[l]) ? "\n" : sep);
	}
}



int main_show(int argc, char* argv[])
{
	bool meta = false;
	int showdim = -1;
	const char* sep = strdup("\t");
	const char* fmt = strdup("%+e%+ei");

	const struct opt_s opts[] = {

		OPT_SET('m', &meta, "show meta data"),
		OPT_INT('d', &showdim, "dim", "show size of dimension"),
		OPT_STRING('s', &sep, "sep", "use <sep> as the separator"),
		OPT_STRING('f', &fmt, "format", "use <format> as the format. Default: \"\%+e\%+ei\""),
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

	print_cfl(N, dims, data, fmt,  sep);

out:
	unmap_cfl(N, dims, data);
	xfree(sep);
	xfree(fmt);
	exit(0);
}



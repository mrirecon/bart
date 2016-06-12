/* Copyright 2014. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <assert.h>
#include <stdio.h>
#include <stdbool.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/opts.h"


static const char usage_str[] = "-b <bitmask> | <dim1> ... <dimN>";
static const char help_str[] = "Convert between a bitmask and set of dimensions.";




int main_bitmask(int argc, char* argv[])
{
	bool inverse = false;
	long flags = 0;

	const struct opt_s opts[] = {

		OPT_SET('b', &inverse, "dimensions from bitmask"),
	};

	cmdline(&argc, argv, 0, 1000, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if ((2 != argc) && inverse)
		error("exactly one argument needed.\n");


	if (!inverse) {

		for (int i = 1; i < argc; i++) {

			int d = atoi(argv[i]);
			assert(d >= 0);

			flags = MD_SET(flags, d);
		}

		printf("%ld\n", flags);

	} else {

		int i = 0;
		flags = atoi(argv[1]);

		while (flags) {

			if (flags & 1)
				printf("%d ", i);

			flags >>= 1;
			i++;
		}

		printf("\n");
	}

	exit(0);
}



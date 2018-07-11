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
	return in_mem_bitmask_main(argc, argv, NULL);
}


int in_mem_bitmask_main(int argc, char* argv[], char* output)
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

		if (output != NULL) {
			safeneg_snprintf(output, 512, "%ld", flags);
		}
		else {
			printf("%ld\n", flags);
		}

	} else {

		int idx = 0;
		int max_length = 512;
		int i = 0;
		flags = atoi(argv[1]);

		while (flags) {

			if (flags & 1) {
				if (output != NULL) {
					idx += safeneg_snprintf(output + idx, max_length - idx, "%d ", i);
				}
				else {
					printf("%d ", i);
				}
			}
			flags >>= 1;
			i++;
		}

		if (output == NULL) {
			printf("\n");
		}
	}

	return 0;
}



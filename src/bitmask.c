/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>

#include "num/multind.h"


static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s -b <bitmask> | <dim1> ... <dimN>\n", name);
}

static void help(void)
{
	printf(	"\nCompute bitmask for specified dimensions.\n");
}


int main_bitmask(int argc, char* argv[])
{
	bool inverse = false;
	unsigned int flags = 0;

	int c;
	while (-1 != (c = getopt(argc, argv, "hb:"))) {

		switch (c) {

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		case 'b':
			flags = atoi(optarg);
			inverse = true;
			break;

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if ((argc - optind < 1) && !inverse) {

		usage(argv[0], stderr);
		exit(1);
	}

	if (!inverse) {


		for (int i = optind; i < argc; i++) {

			int d = atoi(argv[i]);
			assert(d >= 0);

			flags = MD_SET(flags, d);
		}

		printf("%d\n", flags);

	} else {

		int i = 0;

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



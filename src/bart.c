/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>

#include "misc/misc.h"
#include "misc/cppmap.h"


#define DECLMAIN(x) \
extern int main_ ## x(int argc, char* argv[]);
MAP(DECLMAIN, MAIN_LIST)
#undef	DECLMAIN

struct {

	int (*main_fun)(int argc, char* argv[]);
	const char* name;

} dispatch_table[] = {

#define DENTRY(x) { main_ ## x, # x },
	MAP(DENTRY, MAIN_LIST)
#undef  DENTRY
	{ NULL, NULL }
};

static void usage(void)
{
	printf("BART. Available commands are:");

	for (int i = 0; NULL != dispatch_table[i].name; i++) {

		if (0 == i % 6)
			printf("\n");

		printf("%-12s", dispatch_table[i].name);
	}

	printf("\n");
}

int main(int argc, char* argv[])
{
	char* bn = basename(argv[0]);

	if (0 == strcmp(bn, "bart")) {

		if (1 == argc) {

			usage();
			exit(1);
		}

		return main(argc - 1, argv + 1);
	}

	for (int i = 0; NULL != dispatch_table[i].name; i++) {

		if (0 == strcmp(bn, dispatch_table[i].name))
			return dispatch_table[i].main_fun(argc, argv);
	}

	error("Unknwon bart command: \"%s\".\n", bn);
	exit(1);
}



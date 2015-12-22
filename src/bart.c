/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2014-2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include <errno.h>

#include "misc/misc.h"
#include "misc/cppmap.h"

#include "main.h"

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

int main_bart(int argc, char* argv[])
{
	char* bn = basename(argv[0]);

	if (0 == strcmp(bn, "bart")) {

		if (1 == argc) {

			usage();
			exit(1);
		}

		char* tpath = getenv("TOOLBOX_PATH");

		if (NULL != tpath) {

			size_t len = strlen(tpath) + strlen(argv[1]) + 2;
			char* cmd = malloc(len);
			size_t r = snprintf(cmd, len, "%s/%s", tpath, argv[1]);
			assert(r < len);

			if (-1 == execv(cmd, argv + 1)) {

				// only if it doesn't exist - try builtin

				if (ENOENT != errno) {

					perror("Executing bart command failed");
					exit(1);
				}

			} else {

				assert(0);
			}

			free(cmd);
		}

		return main_bart(argc - 1, argv + 1);
	}

	for (int i = 0; NULL != dispatch_table[i].name; i++) {

		if (0 == strcmp(bn, dispatch_table[i].name))
			return dispatch_table[i].main_fun(argc, argv);
	}

	fprintf(stderr, "Unknown bart command: \"%s\".\n", bn);
	exit(1);
}



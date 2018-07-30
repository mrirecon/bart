/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include <errno.h>

#include "misc/misc.h"
#include "misc/debug.h"
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
			return 1;
		}

		const char* tpath[] = {
#ifdef TOOLBOX_PATH_OVERRIDE
			getenv("TOOLBOX_PATH"),
#endif
			"/usr/local/lib/bart/commands/",
			"/usr/lib/bart/commands/",
		};

		for (unsigned int i = 0; i < ARRAY_SIZE(tpath); i++) {

			if (NULL == tpath[i])
				continue;

			size_t len = strlen(tpath[i]) + strlen(argv[1]) + 2;

			char cmd[len];
			size_t r = snprintf(cmd, len, "%s/%s", tpath[i], argv[1]);
			assert(r < len);

			if (-1 == execv(cmd, argv + 1)) {

				// only if it doesn't exist - try builtin

				if (ENOENT != errno) {

					perror("Executing bart command failed");
					return 1;
				}

			} else {

				assert(0);
			}
		}

		return main_bart(argc - 1, argv + 1);
	}

	for (int i = 0; NULL != dispatch_table[i].name; i++) {

		if (0 == strcmp(bn, dispatch_table[i].name)) {

			int save = debug_level;

			int ret = error_catcher(dispatch_table[i].main_fun, argc, argv);

			debug_level = save;

			return ret;
		}
	}

	fprintf(stderr, "Unknown bart command: \"%s\".\n", bn);
	return -1;
}



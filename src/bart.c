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
#include "misc/cppmap.h"

#include "main.h"

#ifdef MEMONLY_CFL
#  ifndef FORCE_BUILTIN_COMMANDS
#   define FORCE_BUILTIN_COMMANDS
#  endif /* !FORCE_BUILTIN_COMMANDS */
#endif /* MEMONLY_CFL */

struct {
	
	int (*main_fun)(int argc, char* argv[]);
	const char* name;

} dispatch_table[] = {

#define DENTRY(x) { main_ ## x, # x },
	MAP(DENTRY, MAIN_LIST)
#undef  DENTRY
	{ NULL, NULL }
};

struct {

	int (*main_fun)(int argc, char* argv[], char* out);
	const char* name;

} in_mem_dispatch_table[] = {
	{ in_mem_bitmask_main,  "bitmask"  },
	{ in_mem_estdelay_main, "estdelay" },
	{ in_mem_estdims_main,  "estdims"  },
	{ in_mem_estshift_main, "estshift" },
	{ in_mem_estvar_main,   "estvar"   },
	{ in_mem_nrmse_main,    "nrmse"    },
	{ in_mem_sdot_main,     "sdot"     },
	{ in_mem_show_main,     "show"     },
	{ in_mem_version_main,  "version"  },
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

	printf("\nin-memory commands overrides are:");

	for (int i = 0; NULL != in_mem_dispatch_table[i].name; i++) {

		if (0 == i % 6)
			printf("\n");

		printf("%-12s", in_mem_dispatch_table[i].name);
	}

	printf("\n");
}

int main_bart(int argc, char* argv[])
{
	return in_mem_bart_main(argc, argv, NULL);
}

// if not NULL, output should point to a memory location with at least *512* elements
int in_mem_bart_main(int argc, char* argv[], char* output)
{
	char* bn = basename(argv[0]);
	if (0 == strcmp(bn, "bart")) {

		if (1 == argc) {

			usage();
			return 1;
		}

#ifndef FORCE_BUILTIN_COMMANDS
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
#endif /* !FORCE_BUILTIN_COMMANDS */

		return in_mem_bart_main(argc - 1, argv + 1, output);
	}

	int debug_level_save = debug_level;
	int ret = -1;
		if (output != NULL) {
			for (int i = 0; NULL != in_mem_dispatch_table[i].name; i++) {
				if (0 == strcmp(bn, in_mem_dispatch_table[i].name)) {
					ret = in_mem_dispatch_table[i].main_fun(argc, argv, output);
					bart_exit_cleanup();
					debug_level = debug_level_save;
					return ret;
				}
			}
		}

		for (int i = 0; NULL != dispatch_table[i].name; i++) {

			if (0 == strcmp(bn, dispatch_table[i].name)) {
				ret = dispatch_table[i].main_fun(argc, argv);
				bart_exit_cleanup();
				debug_level = debug_level_save;
				return ret;
			}
		}

		BART_ERR("Unknown bart command: \"%s\".\n", bn);
		bart_exit_cleanup();
		debug_level = debug_level_save;
		return -1;
	}



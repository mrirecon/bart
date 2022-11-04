/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2021. Martin Uecker.
 + Copyright 2018. Damien Nguyen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2014-2021 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include <errno.h>

#ifdef _WIN32
#include "win/fmemopen.h"
#include "win/basename_patch.h"
#endif

#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/version.h"
#include "misc/debug.h"
#include "misc/cppmap.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#ifdef USE_LOCAL_FFTW
#include "fftw3_local.h"
#define MANGLE(name) local_ ## name
#else
#include <fftw3.h>
#define MANGLE(name) name
#endif

#include "main.h"



extern FILE* bart_output;	// src/misc.c


static void bart_exit_cleanup(void)
{
	if (NULL != command_line)
		XFREE(command_line);

	io_memory_cleanup();

	opt_free_strdup();

#ifdef FFTWTHREADS
	MANGLE(fftwf_cleanup_threads)();
#endif
#ifdef USE_CUDA
	cuda_memcache_clear();
#endif
}



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

int main_bart(int argc, char* argv[argc])
{
	char* bn = basename(argv[0]);

	if (0 == strcmp(bn, "bart") || 0 == strcmp(bn, "bart.exe")) {

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

		for (int i = 0; i < (int)ARRAY_SIZE(tpath); i++) {

			if (NULL == tpath[i])
				continue;

			size_t len = strlen(tpath[i]) + strlen(argv[1]) + 2;

			char (*cmd)[len] = xmalloc(sizeof *cmd);
			int r = snprintf(*cmd, len, "%s/%s", tpath[i], argv[1]);

			if (r >= (int)len) {

				perror("Commandline too long");
				return 1;
			}

			if (-1 == execv(*cmd, argv + 1)) {

				// only if it doesn't exist - try builtin

				if (ENOENT != errno) {

					perror("Executing bart command failed");
					return 1;
				}

			} else {

				assert(0);
			}

			xfree(cmd);
		}

		return main_bart(argc - 1, argv + 1);
	}

	unsigned int v[5];
	version_parse(v, bart_version);

	if (0 != v[4])
		debug_printf(DP_WARN, "BART version is not reproducible.\n");

	for (int i = 0; NULL != dispatch_table[i].name; i++)
		if (0 == strcmp(bn, dispatch_table[i].name))
			return dispatch_table[i].main_fun(argc, argv);

	fprintf(stderr, "Unknown bart command: \"%s\".\n", bn);

	return -1;
}



int bart_command(int len, char* buf, int argc, char* argv[])
{
	int save = debug_level;

	if (NULL != buf) {

		buf[0] = '\0';
		bart_output = fmemopen(buf, (size_t)len, "w");
	}

	int ret = error_catcher(main_bart, argc, argv);

	bart_exit_cleanup();

	debug_level = save;

	if (NULL != bart_output) {

#ifdef _WIN32
		rewind(bart_output);
		fread(buf, 1, len, bart_output);
#endif

		fclose(bart_output);	// write final nul
		bart_output = NULL;
	}

	return ret;
}




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
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/version.h"
#include "misc/debug.h"
#include "misc/cppmap.h"

#include "num/multind.h"
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


#ifndef DIMS
#define DIMS 16
#endif

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

typedef int (main_fun_t)(int argc, char* argv[]); 

struct {

	main_fun_t* main_fun;
	const char* name;

} dispatch_table[] = {

#define DENTRY(x) { main_ ## x, # x },
	MAP(DENTRY, MAIN_LIST)
#undef  DENTRY
	{ NULL, NULL }
};

static int find_command_index(int argc, char* argv[argc])
{
	int i = 1;

	for (; i < argc; ++i) {

		for (unsigned int c = 0; c < ARRAY_SIZE(dispatch_table) - 1; ++c) {

			if (0 == strcmp(argv[i], dispatch_table[c].name))
				return i;
		}
	}

	return 1;
}

static const char help_str[] = "BART. command line flags";

static void usage(void)
{
	printf("Usage: bart [-p flags] [-s start1:start2:...startN] [-e end1:end2:...endN] <command> args...\n");
	printf("BART. Available commands are:");

	for (int i = 0; NULL != dispatch_table[i].name; i++) {

		if (0 == i % 6)
			printf("\n");

		printf("%-12s", dispatch_table[i].name);
	}

	printf("\n");
}


static int bart_exit(int err_no, const char* exit_msg)
{
	if (0 != err_no) {

		if (NULL != exit_msg)
			debug_printf(DP_ERROR, "%s\n", exit_msg);
	}

	return err_no;
}


static int parse_bart_opts(int argc, char* argv[argc])
{
	int command_arg = find_command_index(argc, argv);
	int offset = command_arg;

	if (1 == command_arg)
		return offset;
		
	unsigned long flags = 0;
	long param_start[DIMS] = { [0 ... DIMS - 1] = -1 };
	long param_end[DIMS] = { [0 ... DIMS - 1] = -1 };
	
	struct arg_s args[] = {	};

 	struct opt_s opts[] = {

		OPTL_ULONG('l', "loop", &(flags), "flag", "Flag to specify dimensions for looping"),
		OPTL_VECN('s', "start", param_start, "Start index of range for looping (default: 0)"),
		OPTL_VECN('e', "end", param_end, "End index of range for looping (default: start + 1)"),
 	};
	
	cmdline(&command_arg, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	XFREE(command_line);
	opt_free_strdup();

	int nstart = 0;
	int nend = 0;

	for(; nstart < DIMS && -1 != param_start[nstart]; nstart++);
	for(; nend < DIMS && -1 != param_end[nend]; nend++);

	if (0 != nstart && bitcount(flags) != nstart)
		perror("Size of start values does not coincide with number of selected flags!");
	
	if (0 != nend && bitcount(flags) != nend)
		perror("Size of start values does not coincide with number of selected flags!");
		
	if (0 == nstart)
		for (int i = 0; i < bitcount(flags); i++)
			param_start[i] = 0;

	if (0 == nend)
		for (int i = 0; i < bitcount(flags); i++)
			param_end[i] = param_start[i] + 1;

	long offs_size[DIMS] = { [0 ... DIMS - 1] = 0 };
	long loop_dims[DIMS] = { [0 ... DIMS - 1] = 1 };

	for (int i = 0, j = 0; i < DIMS; ++i) {

		if (MD_IS_SET(flags, i)) {

			offs_size[i] = param_start[j];
			loop_dims[i] = param_end[j] - param_start[j];
			j++;
		}
	}

	init_cfl_loop_desc(DIMS, loop_dims, offs_size, flags, 0);
		
	return offset;

}

static int batch_wrapper(main_fun_t* dispatch_func, int argc, char *argv[argc], long pos)
{
	char* thread_argv[argc + 1];
	char* thread_argv_save[argc];

	for(int m = 0; m < argc; m++) {

		thread_argv[m] = strdup(argv[m]);
		thread_argv_save[m] = thread_argv[m];
	}

	thread_argv[argc] = NULL;

	set_cfl_loop_index(pos);
	int ret = (*dispatch_func)(argc, thread_argv);
		
	io_memory_cleanup();

	for(int m=0; m< argc; ++m)
		free(thread_argv_save[m]);

	return ret;
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
				bart_exit(1, NULL);
			}

			if (-1 == execv(*cmd, argv + 1)) {

				// only if it doesn't exist - try builtin

				if (ENOENT != errno) {

					perror("Executing bart command failed");
					return bart_exit(1, NULL);
				}

			} else {

				assert(0);
			}

			xfree(cmd);
		}

		int offset = parse_bart_opts(argc, argv);
		
		return main_bart(argc - offset, argv + offset);
	}
	
	main_fun_t* dispatch_func = NULL;
	for (int i = 0; NULL != dispatch_table[i].name; i++) {

		if (0 == strcmp(bn, dispatch_table[i].name))
			dispatch_func = dispatch_table[i].main_fun;
	}

	unsigned int v[5];
	version_parse(v, bart_version);

	if (0 != v[4])
		debug_printf(DP_WARN, "BART version is not reproducible.\n");

	if (NULL == dispatch_func) {

	fprintf(stderr, "Unknown bart command: \"%s\".\n", bn);
		return bart_exit(-1, NULL);
	}
	
	long start = cfl_loop_worker_id();
	long total = cfl_loop_desc_total();
	
	int final_ret = 0;

	for (long i = start; (i < total) && (0 == final_ret) ; i += cfl_loop_num_workers()) {

		int ret = batch_wrapper(dispatch_func, argc, argv, i);
		
		if (0 != ret) {

			#pragma omp critical (main_end_condition)
			final_ret = ret;
			bart_exit(ret, "Tool exited with error");
		}
	}

	bart_exit_cleanup();

	return final_ret;
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




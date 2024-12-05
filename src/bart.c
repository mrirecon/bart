/* Copyright 2015. The Regents of the University of California.
 * Copyright 2015-2021. Martin Uecker.
 * Copyright 2018. Damien Nguyen.
 * Copyright 2023-2024. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <libgen.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <signal.h>

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
#include "misc/stream.h"

#include "num/mpi_ops.h"
#include "num/multind.h"
#include "num/rand.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

// also check in commands/ subdir at the bart exe location
#define CHECK_EXE_COMMANDS

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

	stream_unmap_all();

#ifdef FFTWTHREADS
	MANGLE(fftwf_cleanup_threads)();
#endif
#ifdef USE_CUDA
	cuda_memcache_clear();
#endif
#ifdef __EMSCRIPTEN__
	wasm_close_fds();
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

static const char help_str[] = "BART. command line flags";

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


static int bart_exit(int err_no, const char* exit_msg)
{
	if (0 != err_no) {

		if (NULL != exit_msg)
			debug_printf(DP_ERROR, "%s\n", exit_msg);

#ifdef USE_MPI
		MPI_Abort(MPI_COMM_WORLD, err_no);
#endif
	}

	return err_no;
}


static void parse_bart_opts(int* argcp, char*** argvp)
{
	int omp_threads = 1;
	unsigned long flags = 0;
	unsigned long pflags = 0;
	long param_start[DIMS] = { [0 ... DIMS - 1] = -1 };
	long param_end[DIMS] = { [0 ... DIMS - 1] = -1 };
	const char* ref_file = NULL;
	bool use_mpi = false;
	bool version = false;
	bool attach = false;

	struct arg_s args[] = { };

	struct opt_s opts[] = {

		OPTL_ULONG('l', "loop", &(flags), "flag", "Flag to specify dimensions for looping"),
		OPTL_ULONG('p', "parallel-loop", &(pflags), "flag", "Flag to specify dimensions for looping and activate parallelization"),
		OPTL_VECN('s', "start", param_start, "Start index of range for looping (default: 0)"),
		OPTL_VECN('e', "end", param_end, "End index of range for looping (default: start + 1)"),
		OPTL_INT('t', "threads", &omp_threads, "nthreads", "Set threads for parallelization"),
		OPTL_INFILE('r', "ref-file", &ref_file, "<file>", "Obtain loop size from reference file/stream"),
		OPTL_SET('M', "mpi", &use_mpi, "Initialize MPI"),
		OPT_SET('S', &mpi_shared_files, "Maps files from each rank (requires shared files system)"),
		OPTL_SET(0, "version", &version, "print version"),
		OPTL_ULONG(0, "random-dims", &cfl_loop_rand_flags, "flags", "vary random numbers along selected dimensions (default: all)"),
		OPT_SET('d', &attach, "(Wait for debugger)"),
	};

	int next_arg = options(argcp, *argvp, "", help_str, ARRAY_SIZE(opts), opts, ARRAY_SIZE(args), args, true);

	if (version)
		debug_printf(DP_INFO, "%s\n", bart_version);

	*argcp -= next_arg;
	*argvp += next_arg;

	if (attach) {

		fprintf(stderr, "PID: %d", getpid());
		raise(SIGSTOP);
	}

#ifndef _OPENMP
	if (omp_threads > 1) {

		debug_printf(DP_WARN, "WARN: Multiple threads requested, but BART compiled without OPENMP support! Ignoring...\n");
		omp_threads = 1;
	}
#endif

	if (0 != flags && 0 != pflags && flags != pflags)
		error("Inconsistent use of -p and -l!\n");

	flags |= pflags;

	if (1 == omp_threads && 0 != pflags)
		omp_threads = 0;

	const char* ompi_str;

	if (NULL != (ompi_str = getenv("OMPI_COMM_WORLD_SIZE"))) {

		unsigned long mpi_ranks = strtoul(ompi_str, NULL, 10);

		if (1 < mpi_ranks)
			use_mpi = true;
	}

	if (use_mpi)
		init_mpi(argcp, argvp);

	if (NULL != ref_file) {

		long ref_dims[DIMS];
		const void* tmp = load_async_cfl(ref_file, DIMS, ref_dims);
		stream_t s = stream_lookup(tmp);

		if (NULL == s) {

			// normal reference file:
			unmap_cfl(DIMS, ref_dims, tmp);

		} else {

			// reference stream:
			// - input is a pipe so don't close.
			// - flags are determined by stream dims.

			if ((0 != flags) || (0 != pflags))
				error("--ref-file is a stream, this is currently incompatible with --loop and --parallel-loop options!\n");

			flags = stream_get_flags(s);
		}

		assert(-1 == param_end[0]);

		for (int i = 0, ip = 0; i < DIMS; i++)
			if (MD_IS_SET(flags, i))
				param_end[ip++] = ref_dims[i];
	}

	opt_free_strdup();

	int nstart = 0;
	int nend = 0;

	for(; nstart < DIMS && -1 != param_start[nstart]; nstart++);
	for(; nend < DIMS && -1 != param_end[nend]; nend++);

	if (0 != nstart && bitcount(flags) != nstart)
		error("Size of start values does not coincide with number of selected flags!\n");

	if (0 != nend && bitcount(flags) != nend)
		error("Size of start values does not coincide with number of selected flags!\n");

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

#ifdef _OPENMP
	if (0 == omp_threads) {

		if (NULL == getenv("OMP_NUM_THREADS"))
			omp_set_num_threads(omp_get_num_procs());

		omp_threads = omp_get_max_threads();
	}
#endif

	omp_threads = MAX(omp_threads, 1);
	omp_threads = MIN(omp_threads, md_calc_size(DIMS, loop_dims));

	if (1 < mpi_get_num_procs())
		omp_threads = 1;

	init_cfl_loop_desc(DIMS, loop_dims, offs_size, flags, omp_threads, 0);
}


static int batch_wrapper(main_fun_t* dispatch_func, int argc, char *argv[argc], long pos)
{
	char* thread_argv[argc + 1];
	char* thread_argv_save[argc];

	for (int m = 0; m < argc; m++) {

		thread_argv[m] = strdup(argv[m]);
		thread_argv_save[m] = thread_argv[m];
	}

	thread_argv[argc] = NULL;

	set_cfl_loop_index(pos);
	num_rand_init(0ULL);

	int ret = (*dispatch_func)(argc, thread_argv);

	io_memory_cleanup();

	for (int m = 0; m < argc; ++m)
		free(thread_argv_save[m]);

	return ret;
}


int main_bart(int argc, char* argv[argc])
{
	#ifdef __EMSCRIPTEN__
	wasm_fd_offset = 0;
	#endif

	char* bn = basename(argv[0]);

	// only skip over initial bart or bart.exe. calling "bart bart" is an error.
	if (0 == strcmp(bn, "bart") || 0 == strcmp(bn, "bart.exe")) {

		if (1 == argc) {

			usage();
			return -1;
		}

		// This advances argv to behind the bart options
		parse_bart_opts(&argc, &argv);

		bn = basename(argv[0]);
	}


	main_fun_t* dispatch_func = NULL;

	for (int i = 0; NULL != dispatch_table[i].name; i++)
		if (0 == strcmp(bn, dispatch_table[i].name))
			dispatch_func = dispatch_table[i].main_fun;

	bool builtin_found = (NULL != dispatch_func);

	if (builtin_found) {

		debug_printf(DP_DEBUG3, "Builtin found: %s\n", bn);

		unsigned int v[5];
		version_parse(v, bart_version);

		if (0 != v[4])
			debug_printf(DP_WARN, "BART version %s is not reproducible.\n", bart_version);

		int final_ret = 0;


		if (cfl_loop_omp()) {

			// gomp does only use a thread pool for non-nested parallelism!
			// Threads are spawned dynamically with a performance penalty for md_functions,
			// if we have an outer parallel region even if it is inactive.

#ifdef USE_CUDA
			cuda_set_stream_level();
#endif

#pragma omp parallel num_threads(cfl_loop_num_workers())
			{
				long start = cfl_loop_worker_id();
				long total = cfl_loop_desc_total();
				long workers = cfl_loop_num_workers();

				for (long i = start; ((i < total) && (0 == final_ret)); i += workers) {

					int ret = batch_wrapper(dispatch_func, argc, argv, i);

					if (0 != ret) {

#pragma omp critical (main_end_condition)
						final_ret = ret;
						bart_exit(ret, "Tool exited with error");
					}
				}
			}

		} else {

			long start = cfl_loop_worker_id();
			long total = cfl_loop_desc_total();
			long workers = cfl_loop_num_workers();

			mpi_signoff_proc(cfl_loop_desc_active() && (mpi_get_rank() >= total));

			for (long i = start; ((i < total) && (0 == final_ret)); i += workers) {

				int ret = batch_wrapper(dispatch_func, argc, argv, i);

				int tag = ((((i + workers) < total) || (0 != ret)) ? 1 : 0);
				mpi_signoff_proc(cfl_loop_desc_active() && (0 == tag));

				if (0 != ret) {

					final_ret = ret;
					bart_exit(ret, "Tool exited with error");
				}
			}
		}

		deinit_mpi();
		bart_exit_cleanup();

		return final_ret;

	} else {
		// could not find any builtin
		// try to find something in commands

		debug_printf(DP_DEBUG3, "No builtin found: %s\n", argv[0]);

#ifdef CHECK_EXE_COMMANDS
		// also check dirname(PATH_TO_BART)/commands/:
		char exe_loc[1024] = {0};
		ssize_t exe_loc_size = ARRAY_SIZE(exe_loc);
		ssize_t rl = readlink("/proc/self/exe", exe_loc, (size_t)exe_loc_size);

		char* exe_dir = NULL;

		if ((-1 != rl) && (exe_loc_size != rl)) {

			// readlink returned without error and did not truncate
			exe_dir = dirname(exe_loc);
			// no need to check for NULL, as in that case, we skip it in the loop below
		}
#endif

		const char* tpath[] = {
#ifdef CHECK_EXE_COMMANDS
			exe_dir,
#endif
			getenv("BART_TOOLBOX_PATH"),
			getenv("TOOLBOX_PATH"), // support old environment variable
			"/usr/local/lib/bart/",
			"/usr/lib/bart/",
		};

		for (int i = 0; i < (int)ARRAY_SIZE(tpath); i++) {

			if (NULL == tpath[i])
				continue;

			size_t len = strlen(tpath[i]) + strlen(bn) + 10 + 1; // extra space for /commands/ and null-terminator

			char (*cmd)[len] = xmalloc(sizeof *cmd);

			int r = snprintf(*cmd, len, "%s/commands/%s", tpath[i], bn);

			if (r >= (int)len) {

				error("Commandline too long\n");
				return bart_exit(1, NULL); // not really needed, error calls abort()
			}

			debug_printf(DP_DEBUG3, "Trying: %s\n", cmd);

			if (-1 == execv(*cmd, argv)) {

				if (ENOENT != errno) {

					error("Executing bart command failed\n");
					return bart_exit(1, NULL); // not really needed, error calls abort()
				}

			} else {

				assert(0); // unreachable
			}

			xfree(cmd);

		}

		fprintf(stderr, "Unknown bart command: \"%s\".\n", bn);

		return bart_exit(-1, NULL);
	}
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



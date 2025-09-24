/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Center Göttingen.
 * Copyright 2022-2024. Institute of Biomedical Imaging. TU Graz.
 * Copyright 2016. Martin Uecker.
 * Copyright 2018. Damien Nguyen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016	Martin Uecker
 * 2013,2015	Jonathan Tamir
 * 2013		Dara Bahri <dbahri123@gmail.com
 * 2017-2018    Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/time.h>
#include <time.h>
#if !defined(__CYGWIN__) && !defined(_WIN32) && !defined(__EMSCRIPTEN__)
#include <execinfo.h>
#endif

#ifdef USE_DWARF
#include <elfutils/libdwfl.h>
#endif //USE_DWARF

#include "num/multind.h"
#include "num/mpi_ops.h"

#include "misc/io.h"
#include "misc/mmio.h"
#include "misc/cppmap.h"
#include "misc/misc.h"

#include "debug.h"

#define STRSIZE 64


// Patrick Virtue's timing code
double timestamp(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL); // more accurate than <time.h>

	return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
}


void dump_cfl(const char* name, int D, const long dimensions[D], const complex float* src)
{
	io_reserve_output(name);

	complex float* out = create_cfl(name, D, dimensions);

	md_copy(D, dimensions, out, src, sizeof(complex float));

	unmap_cfl(D, dimensions, out);
}

void dump_multi_cfl(const char* name, int N, int D[N], const long* dimensions[N], const complex float* x[N])
{
	complex float* args[N];
	create_multi_cfl(name, N, D, dimensions, args);

	for (int i = 0; i < N; i++)
		md_copy(D[i], dimensions[i], args[i], x[i], sizeof(complex float));

	unmap_multi_cfl(N, D, dimensions, args);
}


int debug_level = -1;
bool debug_logging = false;

static const char* level_strings[] = {
#define LSTR(x) [DP_ ## x] = # x,
	MAP(LSTR, ERROR, WARN, INFO, DEBUG1, DEBUG2, DEBUG3, DEBUG4, TRACE, ())
#undef  LSTR
};

static const char* get_level_str(int level)
{
	assert(level >= 0);

	return (level < DP_ALL) ? level_strings[level] : "ALL";
}

static void get_datetime_str(int len, char* datetime_str)
{
	time_t tv = time(NULL);
	struct tm* dt = gmtime(&tv);

	strftime(datetime_str, (size_t)len, "%F %T", dt);
}

#define RESET	"\033[0m"
#define RED	"\033[31m"


#ifdef REDEFINE_PRINTF_FOR_TRACE
#undef debug_printf
#undef debug_vprintf
#endif


void debug_vprintf(int level, const char* fmt, va_list ap)
{
	if (-1 == debug_level) {

		debug_level = DP_INFO;

		char* str = getenv("BART_DEBUG_LEVEL");

		// support old environment variable:
		if (NULL == str)
			str = getenv("DEBUG_LEVEL");

		if (NULL != str) {

			errno = 0;
			long r = strtol(str, NULL, 10);

			if ((errno == 0) && (0 <= r) && (r < 10))
				debug_level = r;

			errno = 0;
		}
	}

	if (0 < mpi_get_rank())
		debug_level = DP_WARN;

	if (level <= debug_level) {

		FILE* ofp = stderr;

		if (debug_logging) {

			char dt_str[STRSIZE];
			get_datetime_str(STRSIZE, dt_str);

			fprintf(ofp, "[%s] [%s] - ", dt_str, get_level_str(level));

		} else {

			char* str = getenv("BART_DEBUG_STREAM");
			const char* cmd = NULL;

			if (NULL != str) {

				errno = 0;
				long r = strtol(str, NULL, 10);

				if ((errno == 0) && (1 <= r))
					cmd = ptr_printf(" (%s)", command_line ?: "bart wrapper");
			}

			if (level < DP_INFO) {

				char rank[16] = { '\0' };

				if (1 < mpi_get_num_procs())
					sprintf(rank, " [Rank %d]", mpi_get_rank());

				fprintf(ofp, "%s%s%s%s: ", (level < DP_INFO ? RED : ""), get_level_str(level), rank, cmd?:"");
			}

			if (NULL != cmd)
				xfree(cmd);
		}

		vfprintf(ofp, fmt, ap);

		if ((!debug_logging) && (level < DP_INFO))
			fprintf(ofp, RESET);

		fflush(ofp);
	}
}


void debug_printf(int level, const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	debug_vprintf(level, fmt, ap);
	va_end(ap);
}


void debug_vprintf_trace(const char* func_name,
			 const char* file,
			 int line,
			 int level,
			 const char* fmt,
			 va_list ap)

{
#ifndef USE_LOG_BACKEND
	(void)func_name;
	(void)file;
	(void)line;
	debug_vprintf(level, fmt, ap);
#else
	char tmp[1024] = { };
	vsnprintf(tmp, 1023, fmt, ap);

	// take care of the trailing newline often present...
	if ('\n' == tmp[strlen(tmp) - 1])
		tmp[strlen(tmp) - 1] = '\0';

	vendor_log(level, func_name, file, line, tmp);
#endif /* USE_LOG_BACKEND */
}


void debug_printf_trace(const char* func_name,
			const char* file,
			int line,
			int level,
			const char* fmt,
			...)
{
	va_list ap;
	va_start(ap, fmt);
	debug_vprintf_trace(func_name, file, line, level, fmt, ap);
	va_end(ap);
}


void debug_backtrace(size_t n)
{
#if !defined(__CYGWIN__) && !defined(_WIN32) && !defined(__EMSCRIPTEN__)
	void* ptrs[n + 1];
	int l = backtrace(ptrs, n + 1);

	if (l > 1)
		backtrace_symbols_fd(ptrs + 1, l - 1, STDERR_FILENO);
#else
	(void)n;
	debug_printf(DP_WARN, "no backtrace on cygwin.");
#endif
}

#ifdef USE_DWARF

enum { BACKTRACE_SIZE = 50 };

static void debug_good_backtrace_file(FILE * stream, int skip)
{
#if !defined(__CYGWIN__) && !defined(_WIN32)
	char* debuginfo_path = NULL;

	Dwfl_Callbacks callbacks = {

		.find_elf = dwfl_linux_proc_find_elf,
		.find_debuginfo = dwfl_standard_find_debuginfo,
		.debuginfo_path = &debuginfo_path,
	};

	Dwfl* dwfl = dwfl_begin(&callbacks);

	if (NULL == dwfl)
		goto err;

	if (0 != dwfl_linux_proc_report(dwfl, getpid()))
		goto err;

	if (0 != dwfl_report_end(dwfl, NULL, NULL))
		goto err;

	void* stack[BACKTRACE_SIZE + 1];

	int stack_size;	// suppress warning
	stack_size = backtrace(stack, BACKTRACE_SIZE + 1);

	for (int i = skip; i < stack_size; ++i) {

		Dwarf_Addr addr = (Dwarf_Addr)stack[i];

		Dwfl_Module* module = dwfl_addrmodule(dwfl, addr);
		const char* name = dwfl_module_addrname(module, addr);

		Dwfl_Line* dwfl_line;
		int line = -1;
		const char* file = NULL;

		if ((dwfl_line = dwfl_module_getsrc(module, addr))) {

			Dwarf_Addr addr2;
			file = dwfl_lineinfo(dwfl_line, &addr2, &line, NULL, NULL, NULL);
		}

		fprintf(stream, "%d: %p %s", i - skip, stack[i], name);

		if (file)
			fprintf(stream, " at %s:%d", file, line);

		fprintf(stream, "\n");
	}

	dwfl_end(dwfl);
	return;

err:
	debug_printf(DP_WARN, "Backtrace failed\n.");
#else
	debug_printf(DP_WARN, "no backtrace on cygwin.");
#endif
}

const char* debug_good_backtrace_string(int skip)
{
	FILE* fp = tmpfile(); 
	debug_good_backtrace_file(fp, skip);	
	rewind(fp);

	char tmp[1024];
	const char* ret = ptr_printf("%s", "");

	while (NULL != fgets(tmp, 1023, fp)) {

		const char* tmp2 = ptr_printf("%s%s", ret, tmp);
		xfree(ret);
		ret = tmp2;
	}

	fclose(fp);

	return ret;
}

void debug_good_backtrace(int skip)
{
	if (mpi_is_main_proc())
		debug_good_backtrace_file(stderr, skip);
}

#endif // USE_DWARF


void debug_trace(const char* fmt, ...)
{
	debug_printf(DP_TRACE, "TRACE %f: ", timestamp());

	va_list ap;
	va_start(ap, fmt);
	debug_vprintf(DP_TRACE, fmt, ap);
	va_end(ap);
}


#ifdef INSTRUMENT
/* The following functions are called when entering or
 * leaving any function, if instrumentation is enabled with:
 * -finstrument-functions -finstrument-functions-exclude-file-list=debug.c
 */
extern void __cyg_profile_func_enter(void *this_fn, void * /*call_site*/)
{
	debug_trace("ENTER %p\n", this_fn);
}

extern void __cyg_profile_func_exit(void *this_fn, void * /*call_site*/)
{
	debug_trace("LEAVE %p\n", this_fn);
}
#endif


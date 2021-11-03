/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2021. Uecker Lab. University Center Göttingen.
 * Copyright 2016. Martin Uecker.
 * Copyright 2018. Damien Nguyen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016	Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013,2015	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2013		Dara Bahri <dbahri123@gmail.com
 * 2017-2018    Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

#define _GNU_SOURCE

#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#if !defined(__CYGWIN__) && !defined(_WIN32)
#include <execinfo.h>
#endif

#include "num/multind.h"

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
	gettimeofday(&tv, 0); // more accurate than <time.h>

	return tv.tv_sec + 1e-6 * tv.tv_usec;
}


void dump_cfl(const char* name, int D, const long dimensions[D], const complex float* src)
{
	io_reserve_output(name);
	complex float* out = create_cfl(name, D, dimensions);

	md_copy(D, dimensions, out, src, sizeof(complex float));

	unmap_cfl(D, dimensions, out);
}

void dump_multi_cfl(const char* name, int N, int D[N], const long* dimensions[N], const _Complex float* x[N])
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

	strftime(datetime_str, len, "%F %T", dt);
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

		char* str = getenv("DEBUG_LEVEL");
		debug_level = (NULL != str) ? atoi(str) : DP_INFO;
	}

	if (level <= debug_level) {

		FILE* ofp = (level < DP_INFO) ? stderr : stdout;

		if (debug_logging) {

			char dt_str[STRSIZE];
			get_datetime_str(STRSIZE, dt_str);

			fprintf(ofp, "[%s] [%s] - ", dt_str, get_level_str(level));

		}
		else {
			if (level < DP_INFO) {
				fprintf(ofp, "%s%s: ", (level < DP_INFO ? RED : ""), get_level_str(level));
			}
		}

		vfprintf(ofp, fmt, ap);

		if ((!debug_logging) && (level < DP_INFO)) {
			fprintf(ofp, RESET);
		}

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
	UNUSED(func_name); UNUSED(file); UNUSED(line);
	debug_vprintf(level, fmt, ap);
#else
	char tmp[1024] = { 0 };
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
#if !defined(__CYGWIN__) && !defined(_WIN32)
	void* ptrs[n + 1];
	size_t l = backtrace(ptrs, n + 1);

	if (l > 1)
		backtrace_symbols_fd(ptrs + 1, l - 1, STDERR_FILENO);
#else
	UNUSED(n);
	debug_printf(DP_WARN, "no backtrace on cygwin.");
#endif
}



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
extern void __cyg_profile_func_enter(void *this_fn, void *call_site)
{
	UNUSED(call_site);
	debug_trace("ENTER %p\n", this_fn);
}

extern void __cyg_profile_func_exit(void *this_fn, void *call_site)
{
	UNUSED(call_site);
	debug_trace("LEAVE %p\n", this_fn);
}
#endif

/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2016	Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2013,2015	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com
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
#include <execinfo.h>

#include "num/multind.h"

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
	complex float* out = create_cfl(name, D, dimensions);

	md_copy(D, dimensions, out, src, sizeof(complex float));

	unmap_cfl(D, dimensions, out);
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

		} else
		if (level < DP_INFO)
			fprintf(ofp, "%s%s: ", (level < DP_INFO ? RED : ""), get_level_str(level));

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


void debug_backtrace(size_t n)
{
	void* ptrs[n + 1];
	size_t l = backtrace(ptrs, n + 1);

	if (l > 1)
		backtrace_symbols_fd(ptrs + 1, l - 1, STDERR_FILENO);
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

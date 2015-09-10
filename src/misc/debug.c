/* Copyright 2013-2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013	Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013,2015	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com
 */

#define _GNU_SOURCE

#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <fcntl.h>
#include <complex.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>

#include "num/multind.h"
#include "misc/io.h"
#include "misc/mmio.h"

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

static char* get_level_str(int level, char* level_str)
{
	assert(level >= 0);

	switch (level) {

		case DP_ERROR:
			strcpy(level_str, "ERROR");
			break;

		case DP_WARN:
			strcpy(level_str, "WARN");
			break;

		case DP_INFO:
			strcpy(level_str, "INFO");
			break;

		case DP_DEBUG1:
			strcpy(level_str, "DEBUG1");
			break;

		case DP_DEBUG2:
			strcpy(level_str, "DEBUG2");
			break;

		case DP_DEBUG3:
			strcpy(level_str, "DEBUG3");
			break;

		case DP_DEBUG4:
			strcpy(level_str, "DEBUG4");
			break;

		default:
			strcpy(level_str, "ALL");
			break;

	}

	return level_str;
}

static char* get_datetime_str(char* datetime_str)
{

	time_t tv = time(NULL);
	struct tm* dt = gmtime(&tv);

	strftime(datetime_str, STRSIZE, "%F %T", dt);

	return datetime_str;
}

void debug_vprintf(int level, const char* fmt, va_list ap)
{
	if (-1 == debug_level) {

		char* str = getenv("DEBUG_LEVEL");
		debug_level = (NULL != str) ? atoi(str) : DP_INFO;
	}

	if (level <= debug_level) {

		FILE* ofp = (level < DP_INFO) ? stderr : stdout;

		if (true == debug_logging) {

			char level_str[STRSIZE];
			char dt_str[STRSIZE];

			fprintf(ofp, "[%s] [%s] - ", get_datetime_str(dt_str), get_level_str(level, level_str));
		}

		vfprintf(ofp, fmt, ap);
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

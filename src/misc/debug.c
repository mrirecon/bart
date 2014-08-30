/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013	Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 2013 Dara Bahri <dbahri123@gmail.com
 */

#define _GNU_SOURCE

#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <complex.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>

#include "num/multind.h"
#include "misc/io.h"
#include "misc/mmio.h"

#include "debug.h"



// Patrick Virtue's timing code
double timestamp(void) 
{
	struct timeval tv;
	gettimeofday(&tv, 0);

	return tv.tv_sec + 1e-6 * tv.tv_usec;
}


void dump_cfl(const char* name, int D, const long dimensions[D], const complex float* src)
{
	complex float* out = create_cfl(name, D, dimensions);
	md_copy(D, dimensions, out, src, sizeof(complex float));
	unmap_cfl(D, dimensions, out);
}


int debug_level = -1;


void debug_vprintf(int level, const char* fmt, va_list ap)
{
	if (-1 == debug_level) {

		char* str = getenv("DEBUG_LEVEL");
		debug_level = (NULL != str) ? atoi(str) : DP_INFO;
	}

	if (level <= debug_level) {

		FILE* ofp = (level < DP_INFO) ? stderr : stdout;

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

/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __DEBUG_H
#define __DEBUG_H 1

#include "cppwrap.h"

#include <stdarg.h>
#include <stddef.h>

extern void dump_cfl(const char* name, int D, const long dimensions[__VLA(D)], const _Complex float* x);
extern double timestamp(void);

extern int debug_level;
extern _Bool debug_logging;

enum debug_levels { DP_ERROR, DP_WARN, DP_INFO, DP_DEBUG1, DP_DEBUG2, DP_DEBUG3, DP_DEBUG4, DP_ALL };
extern void debug_printf(int level, const char* fmt, ...);
extern void debug_vprintf(int level, const char* fmt, va_list ap);

extern void debug_backtrace(size_t n);

#include "cppwrap.h"


#endif // __DEBUG_H


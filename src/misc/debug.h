/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * Copyright 2018. Damien Nguyen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __DEBUG_H
#define __DEBUG_H 1

#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>

#include "misc/cppwrap.h"

extern void dump_cfl(const char* name, int D, const long dimensions[__VLA(D)], const _Complex float* x);
extern double timestamp(void);

extern int debug_level;
extern _Bool debug_logging;

enum debug_levels { DP_ERROR, DP_WARN, DP_INFO, DP_DEBUG1, DP_DEBUG2, DP_DEBUG3, DP_DEBUG4, DP_TRACE, DP_ALL };
extern void debug_vprintf_trace(const char* func_name,
				const char* file,
				unsigned int line,
				int level, const char* fmt, va_list ap);
extern void debug_printf_trace(const char* func_name,
			       const char* file,
			       unsigned int line,
			       int level, const char* fmt, ...);

// To get the proper function name, file and line when spoofing UTrace
#define debug_printf(level, ...)					\
     debug_printf_trace(__FUNCTION__, __FILE__, __LINE__, level, __VA_ARGS__)
#define debug_vprintf(level, fmt, ap)					\
     debug_vprintf_trace(__FUNCTION__, __FILE__, __LINE__, level, fmt, ap)

extern void debug_backtrace(size_t n);

extern void debug_trace(const char* fmt, ...);

#define TRACE()	debug_trace("%s:%d %s\n", __FILE__, __LINE__, __func__)


#include "misc/cppwrap.h"


#endif // __DEBUG_H


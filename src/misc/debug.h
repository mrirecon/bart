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

#include "misc/cppwrap.h"

extern void dump_cfl(const char* name, int D, const long dimensions[__VLA(D)], const _Complex float* x);
extern void dump_multi_cfl(const char* name, int N, int D[__VLA(N)], const long* dimensions[__VLA(N)], const _Complex float* x[__VLA(N)]);
extern double timestamp(void);

extern int debug_level;

extern _Bool debug_logging;

enum debug_levels { DP_ERROR, DP_WARN, DP_INFO, DP_DEBUG1, DP_DEBUG2, DP_DEBUG3, DP_DEBUG4, DP_TRACE, DP_ALL };


extern void debug_printf(int level, const char* fmt, ...);
extern void debug_vprintf(int level, const char* fmt, va_list ap);

#ifdef REDEFINE_PRINTF_FOR_TRACE
#define debug_printf(level, ...) \
	debug_printf_trace(__FUNCTION__, __FILE__, __LINE__, level, __VA_ARGS__)
#define debug_vprintf(level, fmt, ap) \
	debug_vprintf_trace(__FUNCTION__, __FILE__, __LINE__, level, fmt, ap)
#endif

extern void debug_printf_trace(const char* func_name,
			       const char* file,
			       int line,
			       int level, const char* fmt, ...);
extern void debug_vprintf_trace(const char* func_name,
				const char* file,
				int line,
				int level, const char* fmt, va_list ap);


#define BART_OUT(...) debug_printf_trace(__FUNCTION__, __FILE__, __LINE__, DP_INFO, __VA_ARGS__)
#define BART_ERR(...) debug_printf_trace(__FUNCTION__, __FILE__, __LINE__, DP_ERROR, __VA_ARGS__)
#define BART_WARN(...) debug_printf_trace(__FUNCTION__, __FILE__, __LINE__, DP_WARN, __VA_ARGS__)


extern void debug_backtrace(size_t n);

extern void debug_trace(const char* fmt, ...);

#define TRACE()	debug_trace("%s:%d %s\n", __FILE__, __LINE__, __func__)


#ifdef USE_LOG_BACKEND
// this function must be provided by a vendor backend
extern void vendor_log(int level,
		const char* func_name,
		const char* file,
		unsigned int line,
		const char* message);
#endif


#include "misc/cppwrap.h"


#endif // __DEBUG_H

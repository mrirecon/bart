/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#ifndef __DEBUG_H
#define __DEBUG_H 1

#ifdef __cplusplus
extern "C" {
#ifndef __VLA
#define __VLA(x)
#endif
#else
#ifndef __VLA
#define __VLA(x) static x
#endif
#endif

#include <stdarg.h>

extern void dump_cfl(const char* name, int D, const long dimensions[__VLA(D)], const _Complex float* x);
extern double timestamp(void);

extern int debug_level;
extern _Bool debug_logging;

enum debug_levels { DP_ERROR, DP_WARN, DP_INFO, DP_DEBUG1, DP_DEBUG2, DP_DEBUG3, DP_DEBUG4, DP_ALL };
extern void debug_printf(int level, const char* fmt, ...);
extern void debug_vprintf(int level, const char* fmt, va_list ap);

#ifdef __cplusplus
}
#endif

#endif // __DEBUG_H


/* Copyright 2017-2018. Damien Nguyen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef ICE_OUT_H_INCLUDED
#define ICE_OUT_H_INCLUDED

#include <stdarg.h>

#ifdef __cplusplus
#  define EXTERN_FUNC extern "C"
#else
#  define EXTERN_FUNC extern
#endif /* __cplusplus */

#ifdef BART_SUPER_DEBUG
#  define SUPER_DEBUG_OUT(...) BART_OUT(__VA_ARGS__)
#else
#  define SUPER_DEBUG_OUT(...) (void)0
#endif /* BART_SUPER_DEBUG */

// First argument must be the fmt!
#define BART_OUT(...)						\
     bart_out(__FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)
// First argument must be the fmt!
#define BART_ERR(...)						\
     bart_err(__FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)
// First argument must be the fmt!
#define BART_WARN(...)						\
     bart_warn(__FUNCTION__, __FILE__, __LINE__, __VA_ARGS__)

EXTERN_FUNC void bart_out(const char* func_name,
			  const char* file,
			  unsigned int line,
			  const char* fmt,
			  ...);
EXTERN_FUNC void bart_err(const char* func_name,
			  const char* file,
			  unsigned int line,
			  const char* fmt,
			  ...);
EXTERN_FUNC void bart_warn(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char* fmt,
			   ...);

EXTERN_FUNC void bart_vout(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char*fmt,
			   va_list ap);
EXTERN_FUNC void bart_verr(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char*fmt,
			   va_list ap);
EXTERN_FUNC void bart_vwarn(const char* func_name,
			    const char* file,
			    unsigned int line,
			    const char*fmt,
			    va_list ap);
#endif //ICE_OUT_H_INCLUDED

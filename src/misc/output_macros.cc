/* Copyright 2017-2018. Damien Nguyen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

#include "output_macros.h"

#include <stdio.h>
#include <stdarg.h>

// =============================================================================

#ifdef USE_LOG_SIEMENS_BACKEND
#  include "UTrace.hh"
#endif /* USE_LOG_SIEMENS_BACKEND */

// =============================================================================

EXTERN_FUNC void bart_out(const char* func_name,
			  const char* file,
			  unsigned int line,
			  const char* fmt,
			  ...)
{
     va_list ap;
     va_start(ap, fmt);
     bart_vout(func_name, file, line, fmt, ap);
     va_end(ap);
}

EXTERN_FUNC void bart_vout(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char* fmt,
			   va_list ap)
{
#ifdef USE_LOG_BACKEND
     char tmp[1024] = {""};
     vsnprintf(tmp, 1024, fmt, ap);	
#  ifdef USE_LOG_SIEMENS_BACKEND
     siemens_log_info(func_name, file, line, tmp);
#  endif /* USE_LOG_SIEMENS_BACKEND */
#else
     char tmp[1024] = {"INFO: "};
     vsnprintf(tmp+6, 1018, fmt, ap);	
     printf("%s\n", tmp);
#endif /* USE_LOG_BACKEND */
}

// -----------------------------------------------------------------------------

EXTERN_FUNC void bart_err(const char* func_name,
			  const char* file,
			  unsigned int line,
			  const char* fmt,
			  ...)
{
     va_list ap;
     va_start(ap, fmt);
     bart_verr(func_name, file, line, fmt, ap);
     va_end(ap);
}

EXTERN_FUNC void bart_verr(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char* fmt,
			   va_list ap)
{
#ifdef USE_LOG_BACKEND
     char tmp[1024] = {""};
     vsnprintf(tmp, 1024, fmt, ap);	
#  ifdef USE_LOG_SIEMENS_BACKEND
     siemens_log_err(func_name, file, line, tmp);
#  endif /* USE_LOG_SIEMENS_BACKEND */
#else
     char tmp[1024] = {"ERROR: "};
     vsnprintf(tmp+7, 1017, fmt, ap);
     fprintf(stderr, "%s\n", tmp);
#endif /* USE_LOG_BACKEND */
}

// -----------------------------------------------------------------------------

EXTERN_FUNC void bart_warn(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char* fmt,
			   ...)
{
     va_list ap;
     va_start(ap, fmt);
     bart_vwarn(func_name, file, line, fmt, ap);
     va_end(ap);
}

EXTERN_FUNC void bart_vwarn(const char* func_name,
			    const char* file,
			    unsigned int line,
			    const char* fmt,
			    va_list ap)
{
     
#ifdef USE_LOG_BACKEND
     char tmp[1024] = {""};
     vsnprintf(tmp, 1024, fmt, ap);	
#  ifdef USE_LOG_SIEMENS_BACKEND
     siemens_log_warn(func_name, file, line, tmp);
#  endif /* USE_LOG_SIEMENS_BACKEND */
#else
     char tmp[1024] = {"WARN: "};
     vsnprintf(tmp+6, 1018, fmt, ap);
     fprintf(stderr, "%s\n", tmp);
#endif /* USE_LOG_BACKEND */
}


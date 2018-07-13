/* Copyright 2018. Damien Nguyen.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 */

#ifndef UTRACE_H_INCLUDED
#define UTRACE_H_INCLUDED

extern "C" {
     void siemens_log_info(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char* message);
     void siemens_log_warn(const char* func_name,
			   const char* file,
			   unsigned int line,
			   const char* message);
     void siemens_log_err(const char* func_name,
			  const char* file,
			  unsigned int line,
			  const char* message);
}
#endif //UTRACE_H_INCLUDED

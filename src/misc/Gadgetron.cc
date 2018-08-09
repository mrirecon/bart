/* Copyright 2018. Damien Nguyen
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

#ifndef _Bool
#define _Bool bool
#endif
#include "misc/debug.h"

#include "log.h"

// =============================================================================

extern "C"
void vendor_log(int level,
		const char* func_name,
		const char* file,
		unsigned int line,
		const char* message)
{
     char fname[1024] = {'\0'};
     snprintf(fname, 1023, "%s (BART)", file);
     
     char msg[2048] = {'\0'};
     snprintf(msg, 2047, "%s\n", message);
     
     if (-1 == debug_level) {
	  char* str = getenv("DEBUG_LEVEL");
	  debug_level = (NULL != str) ? atoi(str) : DP_INFO;
     }

     if (level <= debug_level) {
	  switch(level) {
	  case DP_ERROR:
	       Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_ERROR, fname, line, msg);
	       break;
	  case DP_WARN:
	       Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_WARNING, fname, line, msg);
	       break;
	  case DP_INFO:
	       Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_INFO, fname, line, msg);
	       break;
	  default:
	       Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_DEBUG, fname, line, msg);
	  }
     }
}

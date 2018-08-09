/* Copyright 2018. Damien Nguyen
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

#include "misc/debug.h"

// Ideally we would copy the bare minimum code from the file below in order to
// compile BART without requiring to include the header from Orchestra...
#include <Orchestra/Common/ReconTrace.h>

// =============================================================================

extern "C"
void vendor_log(int level,
		const char* func_name,
		const char* file,
		unsigned int line,
		const char* message)
{
     TracePointer trace = Trace::Instance();

     // TODO: the calls below definitely need some improvements to differentiate
     //       between debugging level, as well as properly forwarding location
     //       information to the logging backend
     if (-1 == debug_level) {
	  char* str = getenv("DEBUG_LEVEL");
	  debug_level = (NULL != str) ? atoi(str) : DP_INFO;
     }

     if (level <= debug_level) {
	  switch(level) {
	  case DP_ERROR:
	       trace->ConsoleMsg(message);
	       break;
	  case DP_WARN:
	       trace->ConsoleMsg(message);
	       break;
	  case DP_INFO:
	       trace->ConsoleMsg(message);
	       break;
	  default:
	       trace->ConsoleMsg(message);
	  }
     }
}

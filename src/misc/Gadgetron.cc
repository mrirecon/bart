/* Copyright 2018. Damien Nguyen
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

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
     switch(level) {
     case DP_ERROR:
	  Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_ERROR, file, line, message);
	  break;
     case DP_WARN:
	  Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_WARNING, file, line, message);
	  break;
     case DP_INFO:
	  Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_INFO, file, line, message);
	  break;
     default:
	  Gadgetron::GadgetronLogger::instance()->log(Gadgetron::GADGETRON_LOG_LEVEL_DEBUG, file, line, message);
     }
}

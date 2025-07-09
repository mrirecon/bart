/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include "math.h"

#include "seq/gradient.h"

#include "misc.h"


long round_up_GRT(double val_usec)
{
	if (val_usec < 0)
		return 0;

	long lVal_usec = ceil(val_usec);
	long rem = lVal_usec % GRAD_RASTER_TIME;

	if (rem)
		lVal_usec += GRAD_RASTER_TIME - rem;

	return lVal_usec;
}


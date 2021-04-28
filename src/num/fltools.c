/* Copyright 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include <math.h>
#include <stdlib.h>

#include "misc/nested.h"

#include "fltools.h"




void zsort(int N, complex float tmp[N])
{
	NESTED(int, compare, (const void* _a, const void* _b))
	{
		const complex float* a = _a;
		const complex float* b = _b;
		return copysignf(1., (cabsf(*a) - cabsf(*b)));
	};

	qsort(tmp, N, sizeof(complex float), compare);
}



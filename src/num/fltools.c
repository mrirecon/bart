/* Copyright 2021. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "misc/nested.h"
#include "misc/misc.h"

#include "fltools.h"


static int compare(const void* _a, const void* _b)
{
	const complex float* a = _a;
	const complex float* b = _b;
	return copysignf(1., (cabsf(*a) - cabsf(*b)));
}

void zsort(int N, complex float tmp[N])
{

	qsort(tmp, N, sizeof(complex float), compare);
}

complex float zselect(int N, int k, const complex float x[N])
{
	complex float (*p)[N] = xmalloc(sizeof(*p));

	memcpy(*p, x, sizeof *p);

	zsort(N, *p);

	complex float val = (*p)[k];

	xfree(p);

	return val;
}


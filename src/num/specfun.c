/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 *	Martin Uecker <martin.uecker@med.uni-goettingen.de> 
 */

#include <math.h>

#include "misc/misc.h"

#include "num/chebfun.h"

#include "specfun.h"


/* FIXME: improve precision
 * (but should be good enough for our purposes...)
 */

static float coeff_0to8[] = {

	0.143432, 0.144372, 0.147260, 0.152300, 0.159883,
	0.170661, 0.185731, 0.207002, 0.238081, 0.286336,
	0.366540, 0.501252, 0.699580, 0.906853, 1.000000,
};

static float coeff_8toinf[] = {

	0.405687, 0.405664, 0.405601, 0.405494, 0.405349,
	0.405164, 0.404945, 0.404692, 0.404413, 0.404107,
	0.403782, 0.403439, 0.403086, 0.402724, 0.402359,
	0.401995, 0.401637, 0.401287, 0.400951, 0.400631,
	0.400332, 0.400055, 0.399805, 0.399582, 0.399391,
	0.399231, 0.399106, 0.399012, 0.398998, 0.399001
};



/*
 * modified bessel function
 */
double bessel_i0(double x)
{
	if (x < 0.)
		return bessel_i0(-x);

	if (x < 8.)
		return exp(x) * chebeval(x  / 4. - 1., ARRAY_SIZE(coeff_0to8), coeff_0to8);

	return exp(x) * chebeval(16. / x - 1., ARRAY_SIZE(coeff_8toinf), coeff_8toinf) / sqrt(x);
}



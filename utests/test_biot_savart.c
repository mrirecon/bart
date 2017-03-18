/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "num/vec3.h"

#include "simu/biot_savart.h"

#include "utest.h"


static bool test_vec3_ring(void)
{
	unsigned int N = 10;
	vec3_t r[N];
	vec3_t c = { 0., 0., 0. };
	vec3_t n = { 1., 0., 0. };
	vec3_ring(N, r, c, n, 0.33);

	bool ok = true;

	for (unsigned int i = 0; i < N; i++)
		ok &= (1.E-6 > fabsf(0.33 - vec3_norm(r[i])));

	for (unsigned int i = 0; i < N; i++)
		ok &= (0. == vec3_sdot(r[i], n));

	return ok;
}


static bool test_biot_savart(void)
{
	unsigned int N = 100;
	vec3_t r[N];
	vec3_t c = { 0.6, 0.3, 0.1 };
	vec3_t n = { 1., 0., 0. };
	vec3_ring(N, r, c, n, 0.5);
	vec3_t x;
	biot_savart(x, c, N, (const vec3_t*)r);

	vec3_t d;
	vec3_sub(d, x, n);
	return (1.E-3 > vec3_norm(d)); 
}


UT_REGISTER_TEST(test_vec3_ring);
UT_REGISTER_TEST(test_biot_savart);



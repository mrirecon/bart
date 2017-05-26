/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 *
 * Biot-Savart law.
 *
 */

#include <math.h>
#include <assert.h>

#include "misc/misc.h"
#include "num/vec3.h"

#include "biot_savart.h"


typedef float vec3_t[3];
void biot_savart(vec3_t b, const vec3_t r, unsigned int N, const vec3_t curve[static N])
{
	double c = 1. / (4. * M_PI); /* mu_o */

	vec3_clear(b);

	for (unsigned int i = 0; i < N; i++) {

		vec3_t l;
		vec3_sub(l, curve[(i + 1) % N], curve[i]);

		vec3_t d;
		vec3_sub(d, r, curve[i]);

		double n = vec3_norm(d);

		if (0. == n)
			continue;

		vec3_t x;
		vec3_rot(x, l, d);
		vec3_smul(x, x, c / pow(n, 3.));	//saxpy
		vec3_add(b, b, x);
	}
}


void vec3_ring(unsigned int N, vec3_t ring[N], const vec3_t c, const vec3_t n, float r)
{
	assert(1.E-7 > fabsf(1.f - vec3_norm(n)));

	// compute vec orth to n

	vec3_t b1 = { 1., 1., 1. };

	int d = 0;

	for (unsigned int i = 0; i < 3; i++)
		if (fabsf(n[d]) < fabsf(n[i]))
			d = i;

	b1[d] = -(n[0] + n[1] + n[2] - n[d]) / n[d];

	vec3_smul(b1, b1, 1. / vec3_norm(b1));
	assert(1.E-7 > fabsf(1.f - vec3_norm(b1)));
	assert(1.E-7 > fabsf(vec3_sdot(n, b1)));

	vec3_t b2;
	vec3_rot(b2, b1, n);

	assert(1.E-7 > fabsf(1.f - vec3_norm(b2)));
	assert(1.E-7 > fabsf(vec3_sdot(n, b2)));
	assert(1.E-7 > fabsf(vec3_sdot(b1, b2)));


	for (unsigned int i = 0; i < N; i++) {

		float x = sinf(2. * M_PI * i / N);
		float y = cosf(2. * M_PI * i / N);

		vec3_copy(ring[i], c);
		vec3_saxpy(ring[i], ring[i], r * x, b1);
		vec3_saxpy(ring[i], ring[i], r * y, b2);
	}	
}



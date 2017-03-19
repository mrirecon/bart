/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <math.h>

#include "vec3.h"


void vec3_saxpy(vec3_t dst, const vec3_t src1, float alpha, const vec3_t src2)
{
	for (unsigned int i = 0; i < 3; i++)
		dst[i] = src1[i] + alpha * src2[i];
}

void vec3_sub(vec3_t dst, const vec3_t src1, const vec3_t src2)
{
	vec3_saxpy(dst, src1, -1., src2);
}

void vec3_add(vec3_t dst, const vec3_t src1, const vec3_t src2)
{
	vec3_saxpy(dst, src1, +1., src2);
}

void vec3_copy(vec3_t dst, const vec3_t src)
{
	vec3_saxpy(dst, src, 0., src);
}

void vec3_clear(vec3_t dst)
{
	vec3_saxpy(dst, dst, -1., dst);
}

float vec3_sdot(const vec3_t a, const vec3_t b)
{
	float ret = 0.;

	for (unsigned int i = 0; i < 3; i++)
		ret += a[i] * b[i];

	return ret;
}

float vec3_norm(const vec3_t x)
{
	return sqrtf(vec3_sdot(x, x));
}

void vec3_rot(vec3_t dst, const vec3_t src1, const vec3_t src2)
{
	vec3_t tmp;
	tmp[0] = src1[1] * src2[2] - src1[2] * src2[1];
	tmp[1] = src1[2] * src2[0] - src1[0] * src2[2];
	tmp[2] = src1[0] * src2[1] - src1[1] * src2[0];
	vec3_copy(dst, tmp);
}

void vec3_smul(vec3_t dst, const vec3_t src, float alpha)
{
	vec3_saxpy(dst, (vec3_t){ 0., 0., 0. }, alpha, src);
}


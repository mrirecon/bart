/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <complex.h>
#include <float.h>
#include <math.h>

#include "num/fft.h"
#include "num/flpmath.h"

#include "misc/misc.h"

#include "num/chebfun.h"


static void fft1(int N, complex float tmp[N], const complex float ext[N])
{
	fft(1, (long[1]){ N }, 1, tmp, ext);
	md_zsmul(1, (long[1]){ N }, tmp, tmp, 1. / sqrt((double)N));
}

static void ifft1(int N, complex float tmp[N], const complex float ext[N])
{
	ifft(1, (long[1]){ N }, 1, tmp, ext);
	md_zsmul(1, (long[1]){ N }, tmp, tmp, 1. / sqrt((double)N));
}

void chebpoly(int N, float coeff[N], const float val[N])
{
	complex float ext[(N - 1) * 2];
	complex float tmp[(N - 1) * 2];

	for (int i = 0; i < N - 1; i++)
		ext[i] = val[i];

	for (int i = 0; i < N - 1; i++)
		ext[N - 1 + i] = val[N - 1 - i];

	fft1((N - 1) * 2, tmp, ext);

	for (int i = 0; i < N; i++)
		coeff[i] = crealf(tmp[i]) * sqrt((N - 1) * 2) / (N - 1);

	// strange?
	coeff[0] /= 2.;
	coeff[N - 1] /= 2.;
}




void chebinv(int N, float val[N], const float coeff[N])
{
	complex float ext[(N - 1) * 2];
	complex float tmp[(N - 1) * 2];

	for (int i = 0; i < N; i++)
		ext[i] = coeff[i] / sqrt((N - 1) * 2) * (N - 1);

	for (int i = N; i < (N - 1) * 2; i++)
		ext[i] = 0.;

	ifft1((N - 1) * 2, tmp, ext);

	for (int i = 0; i < N; i++)
		val[i] = crealf(tmp[i]) * 2.;
}


static void resample(int A, int B, float dst[A], const float src[B])
{
	for (int i = 0; i < A; i++)
		dst[i] = 0.;

	for (int i = 0; i < B; i++)
		dst[i % A] = src[i];
}

float chebeval(float x, int N, const float pval[N])
{
	float norm = 0.;
	float val = 0.;

	for (int i = 0; i < N; i++) {

		float dist = x - cosf(M_PI * (float)i / (float)(N - 1));

		if (0. == dist)
			return pval[i];

		float weight = ((0 == i % 2) ? 1. : -1.) / dist;

		if ((0 == i) || (N - 1 == i))
			weight /= 2.;

		norm += weight;
		val += weight * pval[i];
	}

	return val / norm;
}



void chebadd(int A, int B, float dst[(A > B) ? A : B], const float src1[A], const float src2[B])
{
	int N = (A > B) ? A : B;

	float tmp1[N];
	float tmp2[N];

	resample(N, A, tmp1, src1);
	resample(N, B, tmp2, src2);

	for (int i = 0; i < N; i++)
		dst[i] = tmp1[i] + tmp2[i];
}



void chebmul(int A, int B, float dst[A + B], const float src1[A], const float src2[B])
{
	int N = A + B;

	float tmp1[N];
	float tmp2[N];

	resample(N, A, tmp1, src1);
	resample(N, B, tmp2, src2);

	float val1[N];
	float val2[N];

	chebinv(N, val1, tmp1);
	chebinv(N, val2, tmp2);

	for (int i = 0; i < N; i++)
		val1[i] *= val2[i];

	chebpoly(N, dst, val1);
}





float chebint(int N, const float coeff[N])
{
	double sum = 0.;

	for (int i = 0; i < N; i += 2)
		sum += coeff[i] * 2. / (1. - (float)(i * i));

	return sum;
}

void chebindint(int N, float dst[N + 1], const float src[N])
{
	for (int i = 0; i < N + 1; i++)
		dst[i] = 0;

	dst[1] += src[0];
	dst[2] += src[1] / 4.;

	for (int i = 2; i < N; i++) {

		dst[i - 1] -= src[i] / (2. * (i - 1));
		dst[i + 1] += src[i] / (2. * (i + 1));
	}
}

void chebdiff(int N, float dst[N - 1], const float src[N])
{
	dst[N - 2] = 2. * (N - 1) * src[N - 1];
	dst[N - 3] = 2. * (N - 2) * src[N - 2];

	for (int i = N - 4; i > 0; i--)
		dst[i] = dst[i + 2] + 2. * (i + 1) * src[i + 1];

	dst[0] = dst[2] / 2. + src[1];
}

void chebfun2(int N, float coeff[N], float (*fun)(float x))
{
	float val[N];

	for (int i = 0; i < N; i++)
		val[i] = fun(cosf(M_PI * (float)i / (float)(N - 1)));

	chebpoly(N, coeff, val);
}


float* chebfun(int* NP, float (*fun)(float x))
{
	int N = 129;

	while(true) {

		float coeff[N];

		chebfun2(N, coeff, fun);

		int maxind = 0;

		for (int i = 0; i < N; i++)
			if (fabsf(coeff[maxind]) < fabsf(coeff[i]))
				maxind = i;

		if (coeff[N - 1] < 2. * FLT_EPSILON * coeff[maxind]) {

			while (   (coeff[N - 1] < 2. * FLT_EPSILON * coeff[maxind])
			       && (coeff[N - 2] < 2. * FLT_EPSILON * coeff[maxind]))
					N -= 2;

			float* out = xmalloc(sizeof(float) * N);

			for (int i = 0; i < N; i++)
				out[i] = coeff[i];

			*NP = N;
			return out;
		}

		N = (N - 1) * 2 + 1;
	}
}





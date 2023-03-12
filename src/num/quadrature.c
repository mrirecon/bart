/* Copyright 2023. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>

#include "misc/nested.h"

#include "quadrature.h"

void quadrature_trapezoidal(int N, const float t[N + 1], int P, float out[P],
		void CLOSURE_TYPE(sample)(float out[P], int i))
{
	for (int p = 0; p < P; p++)
		out[p] = 0.;

	float last[P];

	for (int p = 0; p < P; p++)
		last[p] = 0.;

	for (int i = 0; i < N; i++) {

		// trapezoidal rule

		float w = t[i + 1] - t[i];

		float n[P];
		NESTED_CALL(sample, (n, i));

		for (int p = 0; p < P; p++) {

			out[p] += w * (n[p] + last[p]) / 2.;
			last[p] = n[p];
		}
	}
}
void quadrature_simpson_ext(int N, float T, int P, float out[P],
		void CLOSURE_TYPE(sample)(float out[P], int i))
{
	assert(10 <= N);

	for (int p = 0; p < P; p++)
		out[p] = 0.;

	// extended Simpson's rule
	// https://mathworld.wolfram.com/Newton-CotesFormulas.html

	for (int i = 0; i < 4; i++) {
	
		float coeff[4] = { 17. / 48., 59 / 48., 43 / 48., 49 / 48. };

		float n1[P];
		sample(n1, i);

		float n2[P];
		sample(n2, N - i);

		for (int p = 0; p < P; p++)
			out[p] += coeff[i] * (n1[p] + n2[p]);
	}

	for (int i = 4; i <= N - 4; i++) {

		float n[P];
		sample(n, i);

		for (int p = 0; p < P; p++)
			out[p] += n[p];
	}

	for (int p = 0; p < P; p++)
		out[p] *= T / N;
}

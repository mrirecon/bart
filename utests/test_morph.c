/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2025 Moritz Blumenthal
 */

#include <complex.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/morph.h"

#include "utest.h"


static bool test_md_center_of_mass(void)
{
	complex float binary[10][10] = {
		{ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 1, 0, 2, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 2, 0, 3, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 3, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 3, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
	};

	enum { N = 4 };
	long dims[N] = { 10, 10, 1, 1 };

	long sdims[N];
	complex float* structure = md_label_simple_connection(N, sdims, 1., 3UL);
	complex float* labels = md_alloc(N, dims, CFL_SIZE);

	long n_labels = md_label(N, dims, labels, &(binary[0][0]), sdims, structure);

	float com[n_labels][N];
	md_center_of_mass(n_labels, N, com, dims, labels, NULL);

	md_free(labels);
	md_free(structure);

	bool ok = (4 == n_labels);

	float expected[4][N] = {
		{ 2.0, 1.0, 0.0, 0.0 },
		{ 4.0, 2.5, 0.0, 0.0 },
		{ 6.0, 4.0, 0.0, 0.0 },
		{ 3.5, 6.5, 0.0, 0.0 },
	};

	for (int i = 0; i < 4; i++) {

		bool found = false;

		for (int j = 0; j < n_labels; j++) {

			bool match = true;
			for (int k = 0; k < N; k++)
				match = match && (expected[i][k] == com[j][k]);

			found = found || match;
		}

		if (!found) {

			print_float(4, expected[i]);
		}

		ok = ok && found;
	}

	return ok;
}


UT_REGISTER_TEST(test_md_center_of_mass);


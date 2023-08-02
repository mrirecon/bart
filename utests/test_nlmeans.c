#include <complex.h>
#include <stdio.h>

#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/rand.h"

#include "num/nlmeans.h"

#include "utest.h"




static bool test_md_znlmeans_distance(void)
{
	const long pdim[] = { 7, 7 };
	// 3x3 array which has been padded to 7x7 by reflection
	const complex float padded[][7] = {
		{ 0, 0,  0, 0, 0,  0, 0 },
		{ 0, 0,  3, 0, 4,  0, 0 },

		{ 0, 3,  3, 0, 4,  0, 0 },
		{ 0, 0,  0, 0, 0,  0, 0 },
		{ 0, 2,  2, 0, 1,  1, 0 },

		{ 0, 0,  2, 0, 1,  0, 0 },
		{ 0, 0,  0, 0, 0,  0, 0 },
	};

	const long odim[] = { 5, 5, 3, 3 };
	// c-array access is transposed compared to md function.
	complex float output[3][3][5][5];

	md_znlmeans_distance(2, pdim, 4, odim, 3, &output[0][0][0][0], &padded[0][0]);

	for (int i = -1; i < 4; i++) {

		for (int j = -1; j < 4; j++) {

			for (int k = -1; k < 2; k++) {

				for (int l = -1; l < 2; l++) {

					//transpos:center[i]  [j]   - center[i+k]  [j+l]   == output[i+1][j+1][k+1][l+1]);
					bool ok = (padded[j + 2][i + 2] - padded[j + l + 2][i + k + 2] == output[l + 1][k + 1][j + 1][i + 1]);

					if (!ok) {

						fprintf(stderr, "Failed at [%d][%d][%d][%d]\n", l, k, j, i);
						return false;
					}
				}
			}
		}
	}

	return true;
}

static bool test_md_znlmeans1(void)
{
	//test if restriction to a search window works as expected
	const long idim[] = { 5, 5 };
	const complex float input[] = {
		0., 0., 0., 0., 0.,
		0., 0., 0., 0., 0.,
		0., 0., 1., 0., 0.,
		0., 0., 0., 0., 0.,
		0., 0., 0., 0., 0.,
	};

	complex float output[5][5];

	md_znlmeans(2, idim, 3, &output[0][0], input, 3, 1, 1, 1);

	complex float x = 0;

	for (int i = 0; i < 5; i++)
		x += cpowf(output[0][i], 2) + cpowf(output[i][0], 2.) + cpowf(output[4][i], 2.) + cpowf(output[i][4], 2.);

	UT_ASSERT(x == 0.);
}

static bool test_md_znlmeans2(void)
{
	//Check invariance of a constant input (the only invariance under NLMeans)
	const long idim[1] = { 10 };
	const complex float input[] = { 1., 1., 1., 1., 1., 1., 1., 1., 1., 1. };
	complex float output[10];

	md_znlmeans(1, idim, 1, output, input, 3, 3, 10, 1.);

	UT_ASSERT(md_znrmse(1, idim, input, output) == 0.);
}

static bool test_md_znlmeans3(void)
{
	//Check successful denoising in one example
	const long idim[1] = { 5 };
	const complex float ref[] = { 1., 0., 0., 0., 1. };
	const complex float input[] = { 1.1, 0., 0., 0., 0.9 };
	complex float output[5];

	md_znlmeans(1, idim, 1, output, input, 3, 5, 0.2, 1.);

	UT_ASSERT(md_znrmse(1, idim, ref, output) < md_znrmse(1, idim, ref, input) / 10.);
}

UT_REGISTER_TEST(test_md_znlmeans_distance);
UT_REGISTER_TEST(test_md_znlmeans1);
UT_REGISTER_TEST(test_md_znlmeans2);
UT_REGISTER_TEST(test_md_znlmeans3);


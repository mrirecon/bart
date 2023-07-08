#include <complex.h>
#include <stdio.h>

#include "utest.h"

#include "num/flpmath.h"
#include "num/rand.h"

#include "num/nlmeans.h"



static bool test_md_zgausspdf(void)
{
/* Comparison against scipy's multivariate_normal.
import scipy.stats
import numpy as np
nX = nY = 11; sigma = 9
X, Y = np.meshgrid(np.arange(nX) - (nX - 1)/2, np.arange(nY) - (nY - 1)/2)
s = scipy.stats.multivariate_normal([0,0], np.array([[sigma,0],[0,sigma]]))
pdf = s.pdf(np.stack((X,Y), axis=-1))
print("complex float x[] = {", end='')
for x in pdf.T:
    print("")
    for i,y in enumerate(x):
        print(f"{y},", end='')
print("};")
*/

	const long dim[] = { 11, 11 };
	complex float x[] = {
0.0010995223491546443,0.0018128058846614335,0.002674506149641422,0.0035308637313793117,0.004171222635474596,0.004409515177532113,0.004171222635474596,0.0035308637313793117,0.002674506149641422,0.0018128058846614335,0.0010995223491546443,
0.0018128058846614335,0.0029888116216916687,0.004409515177532109,0.005821410137868695,0.006877183483932814,0.00727006146667224,0.006877183483932814,0.005821410137868695,0.004409515177532109,0.0029888116216916687,0.0018128058846614335,
0.002674506149641422,0.004409515177532109,0.006505536836035465,0.008588562815826544,0.010146188114027383,0.010725816958894883,0.010146188114027383,0.008588562815826544,0.006505536836035465,0.004409515177532109,0.002674506149641422,
0.0035308637313793117,0.005821410137868695,0.008588562815826544,0.011338558692467643,0.013394924378234939,0.01416014619919741,0.013394924378234939,0.011338558692467643,0.008588562815826544,0.005821410137868695,0.0035308637313793117,
0.004171222635474596,0.006877183483932814,0.010146188114027383,0.013394924378234939,0.015824233393775738,0.016728236160121764,0.015824233393775738,0.013394924378234939,0.010146188114027383,0.006877183483932814,0.004171222635474596,
0.004409515177532113,0.00727006146667224,0.010725816958894883,0.01416014619919741,0.016728236160121764,0.017683882565766154,0.016728236160121764,0.01416014619919741,0.010725816958894883,0.00727006146667224,0.004409515177532113,
0.004171222635474596,0.006877183483932814,0.010146188114027383,0.013394924378234939,0.015824233393775738,0.016728236160121764,0.015824233393775738,0.013394924378234939,0.010146188114027383,0.006877183483932814,0.004171222635474596,
0.0035308637313793117,0.005821410137868695,0.008588562815826544,0.011338558692467643,0.013394924378234939,0.01416014619919741,0.013394924378234939,0.011338558692467643,0.008588562815826544,0.005821410137868695,0.0035308637313793117,
0.002674506149641422,0.004409515177532109,0.006505536836035465,0.008588562815826544,0.010146188114027383,0.010725816958894883,0.010146188114027383,0.008588562815826544,0.006505536836035465,0.004409515177532109,0.002674506149641422,
0.0018128058846614335,0.0029888116216916687,0.004409515177532109,0.005821410137868695,0.006877183483932814,0.00727006146667224,0.006877183483932814,0.005821410137868695,0.004409515177532109,0.0029888116216916687,0.0018128058846614335,
0.0010995223491546443,0.0018128058846614335,0.002674506149641422,0.0035308637313793117,0.004171222635474596,0.004409515177532113,0.004171222635474596,0.0035308637313793117,0.002674506149641422,0.0018128058846614335,0.0010995223491546443,};

	complex float y[sizeof(x)];

	md_zgausspdf(2, dim, y, 9);

	UT_ASSERT(UT_TOL > md_znrmse(2, dim, x, y));
}

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

UT_REGISTER_TEST(test_md_zgausspdf);
UT_REGISTER_TEST(test_md_znlmeans_distance);
UT_REGISTER_TEST(test_md_znlmeans1);
UT_REGISTER_TEST(test_md_znlmeans2);
UT_REGISTER_TEST(test_md_znlmeans3);


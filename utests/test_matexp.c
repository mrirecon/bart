#include <math.h>
#include <complex.h>

#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"
#include "num/matexp.h"


#include "utest.h"

// Estimate first column of matrix exponential exp(mat*M_PI/2) and compare to reference [0, -1]
// See: https://doi.org/10.1137/S00361445024180, chapter: 4
static bool test_zmat_exp(void)
{
	complex float A[4][4] =
		{ { -2.009426e-01-3.135391e-01i,	-9.804413e-02-5.290149e-01i,	-1.465604e-01-1.327557e-01i,	-3.571354e-02-3.449125e-01i },
		{ +2.675718e-01-2.585325e-01i,	+2.961588e-01-6.549746e-01i,	+1.498410e-01+1.465904e-01i,	+2.058340e-01+4.640570e-01i },
		{ +1.634422e-01-7.074941e-01i,	+2.720665e-01+7.809345e-01i,	+1.915712e-01+5.950669e-01i,	+2.699349e-01+3.957221e-01i },
		{ -1.969033e-02+1.245791e+00i,	+6.248414e-02+5.335770e-02i,	+6.939255e-02-1.327612e+00i,	+1.513407e-01+2.785532e-01i } };

	complex float B[4][4] =
		{ { 0.84819764-0.18539282i, -0.11706548-0.30736551i,  0.11399705-0.62641011i,  -0.44408591+1.06909206i },
		{ -0.31972607-0.639699i,    1.03003316-0.58003775i,  0.16193767+1.06706749i,   1.15958364-0.11677061i },
		{ -0.38999632-0.32889717i,  0.55946504+0.09971464i,  1.28527928+0.91767506i,   1.0047425 -1.66147664i },
		{ 0.11179327-0.54839162i,  0.29531177+0.59765872i, -0.24484277+0.7198027i,   1.82777655+0.49979895i } };

	complex float C[4][4];

	zmat_exp(4, 1, C, A);

	float err = 0.;

	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			err += powf(cabsf(B[i][j] - C[i][j]), 2.);

	return (err < 1.E-10);
}

UT_REGISTER_TEST(test_zmat_exp);

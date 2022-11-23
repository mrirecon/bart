/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 *	Martin Uecker <martin.uecker@med.uni-goettingen.de> 
 *	Nick Scholand
 */

#include <math.h>

#include "misc/misc.h"

#include "num/chebfun.h"

#include "specfun.h"

#if 0
/* FIXME: improve precision
 * (but should be good enough for our purposes...)
 */

static const float coeff_0to8[] = {

	0.143432, 0.144372, 0.147260, 0.152300, 0.159883,
	0.170661, 0.185731, 0.207002, 0.238081, 0.286336,
	0.366540, 0.501252, 0.699580, 0.906853, 1.000000,
};

static const float coeff_8toinf[] = {

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

#else

//https://www.advanpix.com/2015/11/11/rational-approximations-for-the-modified-bessel-function-of-the-first-kind-i0-computations-double-precision/

static const double coeff_0to775[17] = {
1.0000000000000000000000801e+00,
2.4999999999999999999629693e-01,
2.7777777777777777805664954e-02,
1.7361111111111110294015271e-03,
6.9444444444444568581891535e-05,
1.9290123456788994104574754e-06,
3.9367598891475388547279760e-08,
6.1511873265092916275099070e-10,
7.5940584360755226536109511e-12,
7.5940582595094190098755663e-14,
6.2760839879536225394314453e-16,
4.3583591008893599099577755e-18,
2.5791926805873898803749321e-20,
1.3141332422663039834197910e-22,
5.9203280572170548134753422e-25,
2.0732014503197852176921968e-27,
1.1497640034400735733456400e-29};

/* [7.75,Inf): */
static const double coeff_775toinf[23] = {
 3.9894228040143265335649948e-01,
 4.9867785050353992900698488e-02,
 2.8050628884163787533196746e-02,
 2.9219501690198775910219311e-02,
 4.4718622769244715693031735e-02,
 9.4085204199017869159183831e-02,
-1.0699095472110916094973951e-01,
 2.2725199603010833194037016e+01,
-1.0026890180180668595066918e+03,
 3.1275740782277570164423916e+04,
-5.9355022509673600842060002e+05,
 2.6092888649549172879282592e+06,
 2.3518420447411254516178388e+08,
-8.9270060370015930749184222e+09,
 1.8592340458074104721496236e+11,
-2.6632742974569782078420204e+12,
 2.7752144774934763122129261e+13,
-2.1323049786724612220362154e+14,
 1.1989242681178569338129044e+15,
-4.8049082153027457378879746e+15,
 1.3012646806421079076251950e+16,
-2.1363029690365351606041265e+16,
 1.6069467093441596329340754e+16};


static double poly(double x, int N, const double coef[N])
{
	double fac = 1;
	double ret = 0;

	for (int i = 0; i < N; i++) {

		ret += fac * coef[i];
		fac *= x;
	}

	return ret;
}

double bessel_i0(double x)
{
	if (x < 0.)
		return bessel_i0(-x);

	if (x < 7.75)
		return 1. + 0.25 * x * x * poly(0.25 * x * x, ARRAY_SIZE(coeff_0to775), coeff_0to775);

	return exp(x) /sqrt(x) *  poly(1. / x, ARRAY_SIZE(coeff_775toinf), coeff_775toinf);
}

#endif


static double factorial(int k)
{
	return (0 == k) ? 1 : (k * factorial(k - 1));
}



// approximate sine integral with power series (only for small x)
double Si_power(double x)
{
	int k_max = 10;
	double sum = 0;

	for (int k = 1; k < k_max; k++)
		sum += pow(-1. , (k - 1)) * pow(x, (2 * k - 1)) / ((2 * k - 1) * factorial(2 * k - 1));

	return sum;
}


static double horner(double x, int N, const double coeff[N])
{
	return coeff[0] + ((1 == N) ? 1. : (x * horner(x, N - 1, coeff + 1)));
}



double sinc(double x)
{
	return (0. == x) ? 1. : (sin(x) / x);
}

float sincf(float x)
{
	return (0. == x) ? 1. : (sinf(x) / x);
}

double jinc(double x)
{
	return (0. == x) ? 1. : (2. * j1(x) / x);
}



// Efficient and accurate calculation of Sine Integral using Padé approximants of the convergent Taylor series
// For details see:
// 	Rowe, B., et al.
//	"GALSIM: The modular galaxy image simulation toolkit".
//	Astronomy and Computing. 10: 121. 2015.
//	arXiv:1407.7676
//  		-> Appendix B: Efficient evaluation of the Sine and Cosine integrals

// helper function to calculate Si accurate for large arguments (> 4)
static double Si_help_f(double x)
{
	double num_coeff[] = {
		+1.,
		+7.44437068161936700618e2,
		+1.96396372895146869801e5,
		+2.37750310125431834034e7,
		+1.43073403821274636888e9,
		+4.33736238870432522765e10,
		+6.40533830574022022911e11,
		+4.20968180571076940208e12,
		+1.00795182980368574617e13,
		+4.94816688199951963482e12,
		-4.94701168645415959931e11,
	};

	double denum_coeff[] = {
		+1.,
		+7.46437068161927678031e2,
		+1.97865247031583951450e5,
		+2.41535670165126845144e7,
		+1.47478952192985464958e9,
		+4.58595115847765779830e10,
		+7.08501308149515401563e11,
		+5.06084464593475076774e12,
		+1.43468549171581016479e13,
		+1.11535493509914254097e13,
	};

	double X = 1. / (x * x);
	double num = horner(X, ARRAY_SIZE(num_coeff), num_coeff);
	double denum = horner(X, ARRAY_SIZE(denum_coeff), denum_coeff);

	return num / denum / x;
}




// helper function to calculate Si accurate for large arguments (> 4)
static double Si_help_g(double x)
{
	double num_coeff[] = {
		+1.,
		+8.13595201151686150e2,
		+2.35239181626478200e5,
		+3.12557570795778731e7,
		+2.06297595146763354e9,
		+6.83052205423625007e10,
		+1.09049528450362786e12,
		+7.57664583257834349e12,
		+1.81004487464664575e13,
		+6.43291613143049485e12,
		-1.36517137670871689e12,
	};

	double denum_coeff[] = {
		+1.,
		+8.19595201151451564e2,
		+2.40036752835578777e5,
		+3.26026661647090822e7,
		+2.23355543278099360e9,
		+7.87465017341829930e10,
		+1.39866710696414565e12,
		+1.17164723371736605e13,
		+4.01839087307656620e13,
		+3.99653257887490811e13,
	};


	double X = 1. / (x * x);

	double num = horner(X, ARRAY_SIZE(num_coeff), num_coeff);
	double denum = horner(X, ARRAY_SIZE(denum_coeff), denum_coeff);

	return ((x < 0) ? -1 : 1) * X * (num / denum);
}

static double Si_large_x(double x)
{
	return M_PI / 2. - Si_help_f(x) * cos(x) - Si_help_g(x) * sin(x);
}

double Si(double x)
{
	// Definition of Si_large_x just for x > 0,
	// therefore use Si(-z) = -Si(z)
	// For more information compare:
	// 	Abramowitz, M., & Stegun, I. 1964, Handbook of Mathematical Functions, 5th edn. (New York: Dover)
	// 		-> Chapter 5.2

	return (fabs(x) <= 4) ? Si_power(x) : ((x < 0) ? -1 : 1) * Si_large_x(fabs(x));
}

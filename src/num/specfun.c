/* Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: Martin Uecker, Nick Scholand
 */

#include <math.h>
#include <assert.h>
#include <stdlib.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/chebfun.h"
#include "num/linalg.h"

#include "specfun.h"

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
double bessel_i0_compat(double x)
{
	if (x < 0.)
		return bessel_i0_compat(-x);

	if (x < 8.)
		return exp(x) * chebeval(x  / 4. - 1., ARRAY_SIZE(coeff_0to8), coeff_0to8);

	return exp(x) * chebeval(16. / x - 1., ARRAY_SIZE(coeff_8toinf), coeff_8toinf) / sqrt(x);
}


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
	// 	Abramowitz, M., & Stegun, I. 1964, Handbook of Mathematical Functions, 5th end. (New York: Dover)
	// 		-> Chapter 5.2

	return (fabs(x) <= 4) ? Si_power(x) : ((x < 0) ? -1 : 1) * Si_large_x(fabs(x));
}

// Gamma Function
double gamma_func(double x)
{
	if (x == (int)x) // FIXME: Some implementations set this case just very high: =1e300
		assert(0. < x);

	double out = 0.;
	double eps = 1e-15;

	// Positive integer gamma(n) = (n - 1)!
	if (eps > fabs(x - (int)x)) {

		if (0. < x) {

			out = 1.;

			int m = (int)(x - 1);

			for (int k = 2; k < m + 1; k++)
				out *= k;
		}
	} else { // Solve Gamma(z) = \int_0^{\inf}t^{z-1}e^{-t}dt

		double r = 1.;
		double z = 0.;

		// Extension beyond |x| > 1 using reflection property
		if (fabs(x) > 1.0) {

			z = fabs(x);

			int m = (int)z;

			for (int k = 1; k < m + 1; k++)
				r *= (z - k);

			z -= m;
		} else
			z = x;

		// Rounded Taylor series coefficients Gamma(x+1)^-1 from
		// Luke, Yudell L. Mathematical Functions and Their Approximations,
		// Elsevier Science & Technology, 1975.
		// Chapter I, Table 1.1
		const double a[25] = {
			1.0,			0.5772156649015329e0,	-0.6558780715202538e0,
			-0.420026350340952e-1,	0.1665386113822915e0,	-0.421977345555443e-1,
			-0.96219715278770e-2,	0.72189432466630e-2,	-0.11651675918591e-2,
			-0.2152416741149e-3,	0.1280502823882e-3,	-0.201348547807e-4,
			-0.12504934821e-5,	0.11330272320e-5,	-0.2056338417e-6,
			0.61160950e-8,		0.50020075e-8,		-0.11812746e-8,
			0.1043427e-9,		0.77823e-11,		-0.36968e-11,
			0.51e-12,		-0.206e-13,		-0.54e-14,
			0.12e-14
		};

		// Evaluate Taylor series

		double s = a[24];

		for (int c = 23; c >= 0; c--)
			s = s * z + a[c];

		out = 1. / (s * z);

		// Correct out following reflection property

		if (fabs(x) > 1.0) {

			out *= r;

			if (0. > x)
				out = -1. * M_PI / (x * out * sin(M_PI * x));
		}
	}

	return out;
}


// Real Valued Gauss Hypergeometric Function
// Following:
//	J.W. Pearson, S. Olver, and M.A. Porter
//	Numerical methods for the computation of the confluent and Gauss hypergeometric functions
//	Number Algor 74, 821–866 (2017).
//	https://doi.org/10.1007/s11075-016-0173-0
static double hyp2f1_powerseries(double a, double b, double c, double x)
{
	int maxiter = 2000;

	double eps = 1.0e-15;

	double out = 1.;
	double rj = 0.;
	double sj = 1.;
	double s_jm1 = 0.;

	for (int j = 1; j < maxiter; j++) {

		rj = (a + j - 1.) * (b + j - 1.) / (j * (c + j - 1.));

		sj = sj * rj * x;

		out += sj;

		if (fabs(out - s_jm1) < (fabs(out) * eps))
			break;

		s_jm1 = out;
	}

	return out;
}

double hyp2f1(double a, double b, double c, double x)
{
	double eps = 1e-15;

	double out = 0.;

	// Test for special cases

	// Limited number of special cases is tested here.
	// See:
	// https://github.com/scipy/scipy/blob/main/scipy/special/special/hyp2f1.h
	// for detailed special case management
	if ( (eps > x) && (eps > a) && (eps > b) )
		out = 1.;

	else if ((eps > fabs(1. - x)) && (0. < (c - a - b)))
		out = gamma_func(c) * gamma_func(c - a - b) / (gamma_func(c - a) * gamma_func(c - b));

	else if ((eps > fabs(1. - x)) && (eps > fabs(c - a + b - 1.)))
		out = sqrt(M_PI) * pow(2., -a) * gamma_func(c) /
			(gamma_func(1. + 0.5 * a - b) * gamma_func(0.5 + 0.5 * a));

	else if (1. >= x) {

		if (0. > x) {

			out = hyp2f1_powerseries(a, c - b, c, x / (x - 1.));

			out *= 1. / pow(1. - x, a);

		} else if ((a > c - a) && (b > c - b)) {

			double x00 = pow(1. - x, c - a - b);

			out = hyp2f1_powerseries(c - a, c - b, c, x);
			out *= x00;

		} else
			out = hyp2f1_powerseries(a, b, c, x);
	}
	else
		error("Hyp2f1 function of this case is not implemented.");

	return out;
}

// Orthogonal Polynomials: Evaluation of associated Legendre function first order P_{\lambda}^{\mu}(x)
static double assoc_legendre(double lambda, double mu, double x)
{
	// FIXME: iterative definition faster?
	double scale = 1. / (gamma_func(1. - mu)) * pow((x + 1.) / (x - 1.), mu / 2.);

	return scale * hyp2f1(-lambda, lambda + 1., 1. - mu, 0.5 * (1. - x));
}

double legendre(double lambda, double x)
{
	return assoc_legendre(lambda, 0., x);
	// return hyp2f1(-lambda, lambda + 1, 1., 0.5 * (1 - x)); // Cheaper
}


static int compare(const void* a, const void* b)
{
	int out = 0;

	if (*(double*)a > *(double*)b)
		out = 1;

	else if (*(double*)a < *(double*)b)
		out = -1;

	return out;
}

#ifndef NO_LAPACK
// Compute weights and sample points for Gauss-Legendre quadrature
void roots_weights_gauss_legendre(const int N, double mu0, double roots[N], double weights[N])
{
	double k[N];
	for (int i = 0; i < N; i++)
		k[i] = i;

	double c_band[2][N];
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < N; j++) {

			if ( (0 == i) && (0 < j) )
				c_band[i][j] = k[j] * sqrt(1. / (4. * k[j] * k[j] - 1.));
			else
				c_band[i][j] = 0.;
		}

	double c[N][N];
	mat_band_reorder(N, 2, c, c_band, true);

	// Find eigenvalues
	double ev[N];
	mat_eig_double(N, ev, c);

	qsort(ev, (size_t)N, sizeof(double), compare);

	double y[N];
	double dy[N];
	double r[N];

	double fm[N];
	double log_fm[N];
	double log_dy[N];

	double max_log_fm = 0.;
	double min_log_fm = 0.;
	double max_log_dy = 0.;
	double min_log_dy = 0.;

	for (int i = 0; i < N; i++) {

		// Newton method to improve roots
		y[i] = legendre(N, ev[i]);
		dy[i] = (-1. * N * ev[i] * legendre(N, ev[i]) + N * legendre(N - 1, ev[i])) / (1. - ev[i] * ev[i]);
		r[i] = ev[i] - y[i] / dy[i];

		// Prepare log-normalization to maintain precision
		fm[i] = legendre(N - 1, ev[i]);
		log_fm[i] = log(fabs(fm[i]));
		log_dy[i] = log(fabs(dy[i]));

		// Find extrema
		max_log_fm = (max_log_fm < log_fm[i]) ? log_fm[i] : max_log_fm;
		min_log_fm = (min_log_fm > log_fm[i]) ? log_fm[i] : min_log_fm;

		max_log_dy = (max_log_dy < log_dy[i]) ? log_dy[i] : max_log_dy;
		min_log_dy = (min_log_dy > log_dy[i]) ? log_dy[i] : min_log_dy;
	}

	double w[N];

	for (int i = 0; i < N; i++) {

		// log-normalization
		fm[i] /= exp((max_log_fm + min_log_fm) / 2.);
		dy[i] /= exp((max_log_dy + min_log_dy) / 2.);

		// Calculation of weights
		w[i] = 1. / (fm[i] * dy[i]);
	}

	double sum = 0.;

	for (int i = 0; i < N; i++) {

		// Symmetrize roots and weights, Assumption: sorted EV!
		roots[i] = (r[i] - r[N - 1 - i]) / 2.;
		weights[i] = (w[i] + w[N - 1 - i]) / 2.;

		sum += weights[i];
	}

	// Normalize with integral of the weight over the orthogonal (mu0)
	for (int i = 0; i < N; i++)
		weights[i] *= mu0 / sum;
}
#endif



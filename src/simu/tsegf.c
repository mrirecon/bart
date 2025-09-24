/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <complex.h>
#include <math.h>

#include "tsegf.h"



#if 0
// Til's definitions

complex double lambda1(complex double z, double k2)
{
	return 1. + z * k2;
}

complex double lambda2(complex double z, double k1, double k2, double cosa)
{
	return 1. - z * (k1 + k2) * cosa + cpow(z, 2.) * k1 * k2;
}

complex double lambda3(complex double z, double k2)
{
	return -1. + z * k2;
}

complex double lambda4(complex double z, double k1, double k2, double cosa)
{
	return -1. + z * (k1 - k2) * cosa + z * z * k1 * k2;
}

complex double gf(complex double z, double k1, double k2, double cosa)
{
	return 0.5 * (1. + csqrt((lambda1(z, k2) * lambda2(z, k1, k2, cosa)) 
	                         / (lambda3(z, k2) * lambda4(z, k1, k2, cosa))));
}
#endif


static complex double tse_gfB(complex double z, double k1, double k2, double cosa)
{
	double u = (1. - cosa) * k2 - cosa * k1;
	double v = (1. - cosa) * k1 - cosa * k2;

	return 1. + z * (u + z * k2 * (v + z * k2 * k1));
}

static complex double tse_DgfB_k1(complex double z, double k2, double cosa)
{
	double u = -cosa;
	double v = (1. - cosa);

	return z * (u + z * k2 * (v + z * k2));
}

static complex double tse_DgfB_k2(complex double z, double k1, double k2, double cosa)
{
	double u = (1. - cosa);
	double v = (1. - cosa) * k1 - cosa * 2. * k2;

	return z * (u + z * (v + z * 2. * k2 * k1));
}

static complex double tse_DgfB_ca(complex double z, double k1, double k2)
{
	double u = -k2 - k1;
	double v = -k1 - k2;

	return z * (u + z * k2 * (v));
}

static complex double tse_gf_sqrt(complex double z, double k1, double k2, double cosa)
{
	return csqrt(tse_gfB(z, k1, k2, cosa) / tse_gfB(z, k1, -k2, cosa));
}

// generating function model for train of (imperfect) spin echos

complex double tse_gf(complex double z, double k1, double k2, double cosa)
{
	return 0.5 * (1. + tse_gf_sqrt(z, k1, k2, cosa));
}

// x * f(y)   \x = f(y)   \y x \y f

// partial derivatives


complex double tse_Dgf_k1(complex double z, double k1, double k2, double cosa)
{
	return 0.25 / tse_gf_sqrt(z, k1, k2, cosa) 
		* (tse_DgfB_k1(z, k2, cosa) / tse_gfB(z, k1, -k2, cosa) 
		- tse_gfB(z, k1, k2, cosa) * tse_DgfB_k1(z, -k2, cosa) 
		  / cpow(tse_gfB(z, k1, -k2, cosa), 2.));
}

complex double tse_Dgf_k2(complex double z, double k1, double k2, double cosa)
{
	return 0.25 / tse_gf_sqrt(z, k1, k2, cosa)
		* (tse_DgfB_k2(z, k1, k2, cosa) / tse_gfB(z, k1, -k2, cosa) 
		+ tse_gfB(z, k1, k2, cosa) * tse_DgfB_k2(z, k1, -k2, cosa) 
	 	  / cpow(tse_gfB(z, k1, -k2, cosa), 2.));
}

complex double tse_Dgf_ca(complex double z, double k1, double k2, double cosa)
{
	return 0.25 / tse_gf_sqrt(z, k1, k2, cosa)
		* (tse_DgfB_ca(z, k1, k2) / tse_gfB(z, k1, -k2, cosa) 
		- tse_gfB(z, k1, k2, cosa) * tse_DgfB_ca(z, k1, -k2) 
 	          / cpow(tse_gfB(z, k1, -k2, cosa), 2.));
}





static void dft(int N, int M, complex float out[N], const complex float buf[M])
{
	for (int i = 0; i < N; i++) {

		out[i] = 0.;

		for (int j = 0; j < M; j++)
			out[i] += cexpf(-2.i * M_PI * (float)(i * j) / (float)M) * buf[j] / (float)M;
	}
}


static void dftH(int N, int M, complex float out[N], const complex float buf[M])
{
	for (int i = 0; i < N; i++) {

		out[i] = 0.;

		for (int j = 0; j < M; j++)
			out[i] += cexpf(+2.i * M_PI * (float)(i * j) / (float)N) * buf[j] / (float)N;
	}
}



void tse(int N, complex float out[N], int M, const float in[4])
{
	complex float buf[M];

	for (int i = 0; i < M; i++) {

		complex double z = cexp(2.i * M_PI * (double)i / (double)M);

		buf[i] = in[0] * tse_gf(z, in[1], in[2], in[3]);
	}

	dft(N, M, out, buf);
}


void tse_der(int N, complex float out[N], int M,
	const float in[4], const float Din[4])
{
	complex float buf[M];

	for (int i = 0; i < M; i++) {

		complex double z = cexp(2.i * M_PI * (double)i / (double)M);

		buf[i] = 0.;
		buf[i] += Din[0] * tse_gf(z, in[1], in[2], in[3]);
		buf[i] += Din[1] * in[0] * tse_Dgf_k1(z, in[1], in[2], in[3]);
		buf[i] += Din[2] * in[0] * tse_Dgf_k2(z, in[1], in[2], in[3]);
		buf[i] += Din[3] * in[0] * tse_Dgf_ca(z, in[1], in[2], in[3]);
	}

	dft(N, M, out, buf);
}


void tse_adj(int N, float out[4], int M,
	float in[4],
	complex float inb[N])
{
	complex float buf[M];

	dftH(M, N, buf, inb);

	for (int i = 0; i < 4; i++)
		out[i] = 0.;

	for (int i = 0; i < M; i++) {

		complex double z = cexp(2.i * M_PI * (double)i / (double)M);

		out[0] += crealf(buf[i] * conjf(tse_gf(z, in[1], in[2], in[3])));
		out[1] += crealf(buf[i] * conjf(in[0] * tse_Dgf_k1(z, in[1], in[2], in[3])));
		out[2] += crealf(buf[i] * conjf(in[0] * tse_Dgf_k2(z, in[1], in[2], in[3])));
		out[3] += crealf(buf[i] * conjf(in[0] * tse_Dgf_ca(z, in[1], in[2], in[3])));
	}
}




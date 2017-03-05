/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include <math.h>
#include <complex.h>
#include <stdio.h>

#include "simu/tsegf.h"

#include "utest.h"


static bool test_tse_der(const float p[4], const float D[4])
{
	int M = 20;	
	complex float out[M];
	complex float out1[M];
	complex float outd[M];

	int N = 200;

	tse(M, out, N, p);
	tse_der(M, outd, N, p, D);

	double sc = 1. / 100.;

	float Dp[4];

	for (int i = 0; i < 4; i++)
		Dp[i] = p[i] + sc * D[i];

	tse(M, out1, N, Dp);

	double err = 0.;

	for (int j = 0; j < M; j++) 
		err += pow(cabs((out1[j] - out[j]) - sc * outd[j]), 2.);

	// printf("err %e\n", (err / (sc * sc)));

	return (err < sc * sc * 1.E-8);
}


static bool test_tse_der1(void)
{
	double T1 = 500.;
	double T2 =  100.;
	double tau = 10.;

	double k1 = exp(-tau / T1);
	double k2 = exp(-tau / T2);
	double cosa = cos(0.75 * M_PI);

	double Dk1 = 0.01 * k1;
	double Dk2 = 0.00 * k2;
	double Dca = 0.00 * cosa;
	//double cosa = cos(M_PI / 2.);
	//printf("#%f %f %f\n", k1, k2, cosa);

	float p[4] = { 1., k1, k2, cosa };
	float D[4] = { 0., Dk1, Dk2, Dca };

	return test_tse_der(p, D);
}

UT_REGISTER_TEST(test_tse_der1);


static bool test_tse_der2(void)
{
	double T1 = 500.;
	double T2 =  100.;
	double tau = 10.;

	double k1 = exp(-tau / T1);
	double k2 = exp(-tau / T2);
	double cosa = cos(0.75 * M_PI);

	double Dk1 = 0.00 * k1;
	double Dk2 = 0.01 * k2;
	double Dca = 0.00 * cosa;
	//double cosa = cos(M_PI / 2.);
	//printf("#%f %f %f\n", k1, k2, cosa);

	float p[4] = { 1., k1, k2, cosa };
	float D[4] = { 0., Dk1, Dk2, Dca };

	return test_tse_der(p, D);
}

UT_REGISTER_TEST(test_tse_der2);


static bool test_tse_der3(void)
{
	double T1 = 500.;
	double T2 =  100.;
	double tau = 10.;

	double k1 = exp(-tau / T1);
	double k2 = exp(-tau / T2);
	double cosa = cos(0.75 * M_PI);

	double Dk1 = 0.00 * k1;
	double Dk2 = 0.00 * k2;
	double Dca = 0.01 * cosa;
	//double cosa = cos(M_PI / 2.);
	//printf("#%f %f %f\n", k1, k2, cosa);

	float p[4] = { 1., k1, k2, cosa };
	float D[4] = { 0., Dk1, Dk2, Dca };

	return test_tse_der(p, D);
}

UT_REGISTER_TEST(test_tse_der3);


static bool test_tse_adj(void)
{
	int M = 20;	
	complex float outd[M];

	int N = 200;

	double T1 = 500.;
	double T2 =  100.;
	double tau = 10.;

	double k1 = exp(-tau / T1);
	double k2 = exp(-tau / T2);
	double cosa = cos(0.75 * M_PI);

	double Dk1 = 0.00 * k1;
	double Dk2 = 0.00 * k2;
	double Dca = 0.01 * cosa;
	//double cosa = cos(M_PI / 2.);
	//printf("#%f %f %f\n", k1, k2, cosa);

	float p[4] = { 1., k1, k2, cosa };
	float D[4] = { 0., Dk1, Dk2, Dca };

	
	tse_der(M, outd, N, p, D);

	float p2[4]; // <A D, l> = <p, A^H l>
	complex float out3[M];

	for (int i = 0; i < M; i++) 
		out3[i] = exp(-(double)i);
	
	tse_adj(M, p2, N, p, out3);


	complex float sc1 = 0.;
	for (int i = 0; i < M; i++) 
		sc1 += conjf(out3[i]) * outd[i];

	float sc2 = 0.;
	for (int i = 0; i < 4; i++) 
		sc2 += p2[i] * D[i];
	
	return (fabsf(crealf(sc1) - sc2) < 1.E-9);
}

UT_REGISTER_TEST(test_tse_adj);


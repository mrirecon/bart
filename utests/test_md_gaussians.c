/* Copyright 2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <complex.h>
#include <stdio.h>

#include "num/gaussians.h"

#include "utest.h"

// all md_gaussian functions are called in md_gaussian_score() -> only test the score function

static bool test_md_gaussian_score_1d(complex float s)
{
    long dims_score[4] = { 1, 1, 1, 1 };
    long dims_x[4]     = { 1, 1, 1, 1 };
    long dims_mu[4]    = { 1, 1, 1, 1 };
    long dims_vars[4]  = { 1, 1, 1, 1 };
    long dims_ws[4]    = { 1, 1, 1, 1 };

    complex float x[1]    = { 0.3 };
    complex float mu[1]   = { 0. };
    complex float vars[1] = { s };
    complex float ws[1]   = { 1. };
    complex float score[1];

    md_gaussian_score(4, dims_score, score, dims_x, x, dims_mu, mu, dims_vars, vars, dims_ws, ws);

    // Analytical gradient: -2 * x / s

    if (cabsf(score[0] - (- 2.f * x[0] / s)) > 1.E-6)
        return false;

    return true;
}

static bool test_md_gaussian_score_2d(complex float s)
{
    long dims_score[4] = { 2, 1, 1, 1 };
    long dims_x[4]     = { 2, 1, 1, 1 };
    long dims_mu[4]    = { 2, 1, 1, 1 };
    long dims_vars[4]  = { 1, 1, 1, 1 }; // only one variance implementation
    long dims_ws[4]    = { 1, 1, 1, 1 };

    complex float x[2]    = { 0.5 - 0.1i, 0.5 + 0.1i };
    complex float mu[2]   = { 0.3, 1.2 };
    complex float vars[1] = { s };
    complex float ws[1]   = { 1. };
    complex float score[2];

    md_gaussian_score(4, dims_score, score, dims_x, x, dims_mu, mu, dims_vars, vars, dims_ws, ws);

    // Check each dimension
    if (cabsf(x[0] - mu[0] +  0.5f * s * score[0]) > 1.E-6) 
        return false;
    if (cabsf(x[1] - mu[1] +  0.5f * s * score[1]) > 1.E-6) 
        return false;

    return true;
}

static bool test_md_gaussian_score_4d(complex float s)
{
    long dims_score[4] = { 2, 2, 1, 1 };
    long dims_x[4]     = { 2, 2, 1, 1 };
    long dims_mu[4]    = { 2, 2, 1, 1 };
    long dims_vars[4]  = { 1, 1, 1, 1 }; // only one variance implementation
    long dims_ws[4]    = { 1, 1, 1, 1 };

    complex float x[4] = { 0.9 - 0.5i, 0.5 + 0.2i, 0.7 - 0.9i, 0.8 + 0.3i };
    complex float mu[4] = { 0 + 0.5i, 0.9 - 0.9i, 0.2 + 0.1i, 0.2 + 0.i };
    complex float vars[1] = { s };
    complex float ws[1]   = { 1. };
    complex float score[4];

    md_gaussian_score(4, dims_score, score, dims_x, x, dims_mu, mu, dims_vars, vars, dims_ws, ws);

    // Check each element in the 2x2 array
    for (int i = 0; i < 4; i++) {
        if (cabsf(x[i] - mu[i] +  0.5f * s * score[i]) > 1.E-6) // real valued variance
            return false;
    }

    return true;
}

static bool test_md_gaussian_score_multisamples(complex float s)
{
    long dims_score[4] = { 2, 1, 2, 1 };
    long dims_x[4]     = { 2, 1, 2, 1 };
    long dims_mu[4]    = { 2, 1, 1, 1 };
    long dims_vars[4]  = { 1, 1, 1, 1 }; // only one variance implementation
    long dims_ws[4]    = { 1, 1, 1, 1 };

    complex float x[4]    = { 0.8 - 0.3i, 0.2 + 0.9i, 0.5 - 0.6i, 0.9 + 0.5i  };
    complex float mu[2]   = { 0.3, 1.2 };
    complex float vars[1] = { s };
    complex float ws[1]   = { 1. };
    complex float score[4];

    md_gaussian_score(4, dims_score, score, dims_x, x, dims_mu, mu, dims_vars, vars, dims_ws, ws);

    // Check each dimension
    if (cabsf(x[0] - mu[0] +  0.5f * s * score[0]) > 1.E-6) // real valued variance
        return false;
    if (cabsf(x[1] - mu[1] +  0.5f * s * score[1]) > 1.E-6) 
        return false;
    if (cabsf(x[2] - mu[0] +  0.5f * s * score[2]) > 1.E-6) 
        return false;
    if (cabsf(x[3] - mu[1] +  0.5f * s * score[3]) > 1.E-6)
        return false;

    return true;
}

static bool test_md_gaussian_score_multigauss(complex float s)
{
    long dims_score[4] = { 2, 1, 1, 1 };
    long dims_x[4]     = { 2, 1, 1, 1 };
    long dims_mu[4]    = { 2, 1, 1, 2 };
    long dims_vars[4]  = { 1, 1, 1, 2 };
    long dims_ws[4]    = { 1, 1, 1, 2 };

    complex float x[2]    = { 0.8 - 0.3i, 0.2 + 0i };
    complex float mu[4]   = { 0, 0, 0, 0 }; // mean = 0
    complex float vars[2] = { s, s }; // sigma = sigma1 = sigma2
    complex float ws[2]   = { 0.5, 0.5 }; // w = w1 = w2
    complex float score[2];

    md_gaussian_score(4, dims_score, score, dims_x, x, dims_mu, mu, dims_vars, vars, dims_ws, ws);

    // Check each dimension
    // Analytical: score = 2 * x / sigma^2
    if (cabsf(score[0] +  2.f * (x[0] / vars[0])) > 1.E-6)
        return false;
    if (cabsf(score[1] +  2.f * (x[1] / vars[1])) > 1.E-6)
        return false;

    return true;
}


static bool test_md_gaussian_score_2d_both(complex float s)
{
	long dims_score[4] = { 2, };
	long dims_x[4]     = { 2, };
	long dims_mu[4]    = { 2, };
	long dims_vars[4]  = { 1, }; // only one variance implementation
	long dims_ws[4]    = { 1, };

	complex float x[2]    = { 0.5 - 0.1i, 0.5 + 0.1i };
	complex float mu[2]   = { 0.3, 1.2 };
	complex float vars[1] = { s };
	complex float ws[1]   = { 1. };
	complex float score[2];
	complex float score2[2];

	complex float visqrt[2][2] = {
		{ powf(s, -1.), 0. },
		{ 0., powf(s, -1.) },
	};

	md_gaussian_score(1, dims_score, score, dims_x, x, dims_mu, mu, dims_vars, vars, dims_ws, ws);
	gaussian_score(2, mu, visqrt, x, score2);

	if (cabsf(score[0] - score2[0]) > 1.E-6)
		return false;

	if (cabsf(score[1] - score2[1]) > 1.E-6)
		return false;

	return true;
}


static bool test_md_gaussian_score()
{
    for (float s = 0.1; s < 1.3; s += 0.1)
        if (!test_md_gaussian_score_1d(s))
            return false;

    for (float s = 0.1; s < 1.3; s += 0.1)
        if (!test_md_gaussian_score_2d(s))
            return false;

    for (float s = 0.1; s < 1.3; s += 0.1)
        if (!test_md_gaussian_score_2d_both(s))
            return false;

    for (float s = 0.1; s < 1.3; s += 0.1)
        if (!test_md_gaussian_score_4d(s))
            return false;

    for (float s = 0.1; s < 1.3; s += 0.1)
        if (!test_md_gaussian_score_multigauss(s))
            return false; 

    for (float s = 0.1; s < 1.3; s += 0.1)
        if (!test_md_gaussian_score_multisamples(s))
            return false;

    return true;
}

UT_REGISTER_TEST(test_md_gaussian_score);




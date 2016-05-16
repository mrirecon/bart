/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

#include "simplex.h"


/*
 * transform matrix so that (d, n) = 1 and (:, n) = 0
 */
static void trafo(unsigned int D, unsigned int N, float A[D][N], unsigned int d, unsigned int n)
{
	float mul = A[d][n];

	for (unsigned int k = 0; k < N; k++)
		A[d][k] /= mul;

	for (unsigned int l = 0; l < D; l++) {

		if (l != d) {

			mul = A[l][n];

			for (unsigned int k = 0; k < N; k++)
				A[l][k] -= mul * A[d][k];
		}
	}			
}




static bool feasible_p(unsigned int D, unsigned int N, const float x[N], /*const*/ float A[D + 1][N + 1])
{
	bool ok = true;

	for (unsigned int j = 0; j < D; j++) {

		float sum = 0.;

		for (unsigned int i = 0; i < N; i++)
			sum += A[1 + j][i] * x[i];

		ok &= (0 == A[1 + j][N] - sum);
	}

	return ok;
}

static void solution(unsigned int D, unsigned int N, float x[N], /*const*/ float A[D + 1][N + 1])
{
	// this is needed to deel with duplicate columns
	bool used[D];
	for (unsigned int i = 0; i < D; i++)
		used[i] = false;

	for (unsigned int i = 0; i < N; i++) {

		x[i] = -1.;
		int pos = -1;

		for (unsigned int j = 0; j < D; j++) {

			if (0. == A[1 + j][i])
				continue;

			if ((1. == A[1 + j][i]) && (-1. == x[i]) && !used[j]) {

				x[i] = A[1 + j][N];
				pos = j;
				used[j] = true;

			} else {

				x[i] = -1.;	
				break;
			}
		}

		if (-1. == x[i]) { // non-basic

			x[i] = 0.;

			if (-1 != pos)
				used[pos] = false;
		}
	}

	//assert(feasible_p(D, N, x, A));
}

extern void print_tableaux(unsigned int D, unsigned int N, /*const*/ float A[D + 1][N + 1]);
void print_tableaux(unsigned int D, unsigned int N, /*const*/ float A[D + 1][N + 1])
{
	float x[N];
	solution(D, N, x, A);


	float y[D];
	for (unsigned int j = 0; j < D; j++) {

		y[j] = 0.;

		for (unsigned int i = 0; i < N; i++)
			y[j] += A[1 + j][i] * x[i];
	}

	printf("           ");

	for (unsigned int i = 0; i < N; i++)
		printf("x%d    ", i);

	printf("\nSolution: ");
	for (unsigned int i = 0; i < N; i++)
		printf(" %0.2f ", x[i]);

	printf("(%s)\n", (feasible_p(D, N, x, A)) ? "feasible" : "infeasible");
	printf("      Max ");

	for (unsigned int i = 0; i < N; i++)
		printf("%+0.2f ", A[0][i]);

	printf("  %+0.2f s.t.:\n", A[0][N]);
	for (unsigned int j = 0; j < D; j++) {

		printf("          ");

		for (unsigned int i = 0; i < N; i++)
			printf("%+0.2f ", A[1 + j][i]);

		printf("= %+0.2f | %+0.2f\n", A[1 + j][N], y[j]);
	}

	printf("Objective: %0.2f\n", A[0][N]);
}




/*
 * maximize c^T x subject to Ax = b and x >= 0
 *
 * inplace, b is last column of A, c first row
 */
static void simplex2(unsigned int D, unsigned int N, float A[D + 1][N + 1])
{
	// 2. Loop over all columns

//	print_tableaux(D, N, A);

	while (true) {

		unsigned int i = 0;

		for (i = 0; i < N; i++)
			 if (A[0][i] < 0.)
				break;

		if (i == N)
			break;

		// 3. find pivot element

		// Bland's rule

		int pivot_index = -1;
		float pivot_value = 0.;

		for (unsigned int j = 1; j < D + 1; j++) {

			if (0. < A[j][i]) {

				float nval = A[j][N] / A[j][i];

				if ((-1 == pivot_index) || (nval < pivot_value)) {

					pivot_value = nval;
					pivot_index = j;
				}
			}
		}

		if (-1 == pivot_index)
			break;

//		printf("PI %dx%d\n", pivot_index, i);

		trafo(D + 1, N + 1, A, pivot_index, i);

//		print_tableaux(D, N, A);
		float x[N];
		solution(D, N, x, A);
		assert(feasible_p(D, N, x, A));
	}
//	print_tableaux(D, N, A);
}

/*
 * maximize c^T x subject to Ax <= b and x >= 0
 */
void (simplex)(unsigned int D, unsigned int N, float x[N], const float c[N], const float b[D], const float A[D][N])
{
	// 1. Step: slack variables
	// max c^T x      Ax + z = b    x,z >= 0

	float A2[D + 1][N + D + 1];

	for (unsigned int i = 0; i < N + D + 1; i++) {

		A2[0][i] = (i < N) ? -c[i] : 0.;

		for (unsigned int j = 0; j < D; j++) {

			if (i < N) 
				A2[1 + j][i] = A[j][i];
			else if (i == N + D)
				A2[1 + j][i] = b[j];
			else
				A2[1 + j][i] = (i - N == j) ? 1. : 0.;
		}
	}

	simplex2(D, N + D, A2);

	// extract results:
	float x2[D + N];
	solution(D, D + N, x2, A2);

	for (unsigned int i = 0; i < N; i++)
		x[i] = x2[i];
}



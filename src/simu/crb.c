/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2021 Volkert Roeloffs
 */

#include <math.h>
#include <complex.h>

#include "num/multind.h"
#include "num/linalg.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "simu/crb.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

void compute_crb(int P, float rCRB[P], complex float A[P][P], int M, int N, const complex float derivatives[M][N], const complex float signal[N], const unsigned long idx_unknowns[P - 1])
{
	// assume first P-1 entries in derivates are w.r.t. parameters (i.e. not M0) 
	assert(P <= M + 1); // maximum 1 + M unknowns 	
 
	// null Fisher information first 
	for (int i = 0; i < P; i++)
		for (int j = 0; j < P; j++) 
			A[i][j] = 0;

	// compute Fisher information matrix
	for (int n = 0; n < N; n++) {

		A[0][0] += conjf(signal[n]) * signal[n];

		for (int i = 1; i < P; i++) {

			A[i][0] += conj(derivatives[idx_unknowns[i - 1]][n]) * signal[n];

			for (int j = 1; j <= i; j++)
				A[i][j] += conjf(derivatives[idx_unknowns[i - 1]][n]) * derivatives[idx_unknowns[j - 1]][n];
		}
	}

	// complete upper triangle with compl. conjugates
	for (int j = 1; j < P; j++)
		for (int i = 0; i < j; i++)
			A[i][j] = conjf(A[j][i]);

	complex float A_inv[P][P];
	mat_inverse(P, A_inv, A);

	for (int i = 0; i < P; i++) 
		rCRB[i] = crealf(A_inv[i][i]);
}

void normalize_crb(int P, float rCRB[P], int N, float TR, float T1, float T2, float B1, float omega, const unsigned long idx_unknowns[P - 1])
{
	UNUSED(omega);

	float normvalues[4];
	normvalues[0] = powf(T1, 2);
	normvalues[1] = powf(T2, 2);
	normvalues[2] = powf(B1, 2);	
	normvalues[3] = 1;

	rCRB[0] *= N * TR; // M0 normalization

	for (int i = 1; i < P; i++) {

		rCRB[i] /= normvalues[idx_unknowns[i - 1]];
		rCRB[i] *= N * TR;
	}
}

void getidxunknowns(int Q, unsigned long idx_unknowns[Q], long unknowns)
{
	int j = 0;

	for (int i = 0; i < 4; i++) {

		if (1 & unknowns) {

			idx_unknowns[j] = i;
			j++;
		}

		unknowns >>= 1;
	}
}

void display_crb(int P, float rCRB[P], complex float fisher[P][P], unsigned long idx_unknowns[P - 1])
{
	bart_printf("Fisher information matrix: \n");

	for (int i = 0; i < P; i++) {

		for(int j = 0; j < P; j++)
			bart_printf("%1.2f%+1.2fi ", crealf(fisher[i][j]), cimagf(fisher[i][j]));

		bart_printf("\n");
	}

	char labels[][40] = { "rCRB T1", "rCRB T2", "rCRB B1", " CRB OF" };

	bart_printf("\n");
	bart_printf("rCRB M0: %3.3f\n", rCRB[0]);

	for(int i = 1; i < P; i++)
		bart_printf("%s: %3.3f\n", labels[idx_unknowns[i - 1]], rCRB[i]);

	bart_printf("\n");
}


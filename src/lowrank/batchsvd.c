/* Copyright 2015-2016. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 *	2015 Frank Ong
 *	2016 Martin Uecker
 */

#include <math.h>
#include <complex.h>

#include "misc/misc.h"

#include "num/blas.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/multind.h"
#include "num/vptr.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "batchsvd.h"



void batch_svthresh(int M, int N, int num_blocks, float lambda, complex float dst[num_blocks][N][M])
{
	bool copy = is_vptr(&dst[0][0][0]);
#ifdef USE_CUDA
	copy = copy || cuda_ondevice(&dst[0][0][0]);
#endif

	if (copy) {

		PTR_ALLOC(complex float[num_blocks][N][M], tmp);
		md_copy(3, MD_DIMS(M, N, num_blocks), &(*tmp)[0][0][0], &dst[0][0][0], sizeof(complex float));

		batch_svthresh(M, N, num_blocks, lambda, *tmp);

		md_copy(3, MD_DIMS(M, N, num_blocks), &dst[0][0][0], &(*tmp)[0][0][0], sizeof(complex float));
		PTR_FREE(tmp);

		return;
	}

#pragma omp parallel
    {
	int minMN = MIN(M, N);

	PTR_ALLOC(complex float[minMN][M], U);
	PTR_ALLOC(complex float[N][minMN], VT);
	PTR_ALLOC(float[minMN], S);
	PTR_ALLOC(complex float[minMN][minMN], AA);

#pragma omp for
	for (int b = 0; b < num_blocks; b++) {

		// Compute upper bound | A^T A |_inf

		// FIXME: this is based on gratuitous guess-work about the obscure
		// API of this FORTRAN from ancient times... Is it really worth it?

		blas_csyrk('U', (N <= M) ? 'T' : 'N', (N <= M) ? N : M, (N <= M) ? M : N, 1., M, dst[b], 0., minMN, *AA);

		// lambda_max( A ) <= max_i sum_j | a_i^T a_j |

		float s_upperbound = 0;

		for (int i = 0; i < minMN; i++) {

			float s = 0;

			for (int j = 0; j < minMN; j++)
				s += cabsf((*AA)[MAX(i, j)][MIN(i, j)]);

			s_upperbound = MAX(s_upperbound, s);
		}

		/* avoid doing SVD-based thresholding if we know from
		 * the upper bound that lambda_max <= lambda and the
		 * result must be zero */

		if (s_upperbound < lambda * lambda) {

			mat_zero(N, M, dst[b]);
			continue;
		}

		lapack_svd_econ(M, N, *U, *VT, *S, dst[b]);

		// soft threshold
		for (int i = 0; i < minMN; i++)
			for (int j = 0; j < N; j++)
				(*VT)[j][i] *= ((*S)[i] < lambda) ? 0. : ((*S)[i] - lambda);


		blas_matrix_multiply(M, N, minMN, dst[b], *U, *VT);
	}

	PTR_FREE(U);
	PTR_FREE(VT);
	PTR_FREE(S);
	PTR_FREE(AA);
    } // #pragma omp parallel
}


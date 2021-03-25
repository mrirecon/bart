/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 */


/* References
 * ----------
 *
 * SSA:
 *
 * Vautard R, Ghil M.
 * Singular spectrum analysis in nonlinear dynamics, with applications to 
 * paleoclimatic time series. Physica D: Nonlinear Phenomena, 1989;35:395-424.
 *
 * (and others)
 *
 * SSA-FARY:
 *
 * Rosenzweig S, Scholand N, Holme HCM, Uecker M.
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular 
 * Spectrum Analysis (SSA-FARY). IEEE Trans. Med. Imag. 2020; in press.
 *
 * General comments:
 *
 * The rank option '-r' allows to "throw away" basis functions:
 *	rank < 0: throw away 'rank' basis functions with high singular values
 *	rank > 0: keep only 'rank' basis functions with the highest singular value
 *
 * The group option '-g' implements what is called 'grouping' in SSA 
 * literature, by selecting EOFs with a bitmask.
 * group < 0: do not use the selected group for backprojection, but all other
 *            EOFs (= filtering)
 * group > 0: use only the selected group for backprojection
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/init.h"
#include "num/casorati.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/flpmath.h"

#include "calib/calib.h"
#include "calib/estvar.h"
#include "calib/calmat.h"

#include "ssa.h"


static bool check_selection(const long group, const int j)
{
	if (j > 30)
		return false; // group has only 32 bits

	return (labs(group) & (1 << j));
}


static void ssa_backprojection( const long N,
				const long M,
				const long kernel_dims[3],
				const long cal_dims[DIMS],
				complex float* back,
				const long A_dims[2],
				const complex float* A,
				const long U_dims[2],
				const complex float* U,
				const complex float* UH,
				const int rank,
				const long group)
{
	assert((N == U_dims[0]) && (N == U_dims[1]));

	// PC = UH @ A
	/* Consider:
	 * AAH = U @ S_square @ UH
	 * A = U @ S @ VH --> PC = S @ VH = UH @ A
	 */
	long PC_dims[2] = { N, M };
	complex float* PC = md_alloc(2, PC_dims, CFL_SIZE);

	long PC2_dims[3] = { N, 1, M };
	long U2_dims[3] = { N, N, 1 };
	long A3_dims[3] = { 1, N, M };

	md_ztenmul(3, PC2_dims, PC, U2_dims, UH, A3_dims, A);

	long kernelCoil_dims[4];

	md_copy_dims(3, kernelCoil_dims, kernel_dims);

	kernelCoil_dims[3] = cal_dims[3];


	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {

			if (rank < 0)
				PC[i * N + j] *= (j >= abs(rank)) ? 1. : 0.;
			else
			if (rank > 0)
				PC[i * N + j] *= (j >= rank) ? 0. : 1.;
			else
			if (group < 0)
				PC[i * N + j] *= (check_selection(group, j)) ? 0. : 1.;
			else
				PC[i * N + j] *= (check_selection(group, j)) ? 1. : 0.;
		}
	}

	// A_LR = U @ PC
	long PC3_dims[3] = { 1, N, M };
	long A4_dims[3] = { N, 1, M };

	complex float* A_backproj = md_alloc(2, A_dims, CFL_SIZE);

	md_ztenmul(3, A4_dims, A_backproj, U2_dims, U, PC3_dims, PC);


	// Reorder & Anti-diagonal summation
	long kern_dims[4];
	md_set_dims(4, kern_dims, 1);
	md_min_dims(4, ~0u, kern_dims, kernelCoil_dims, cal_dims);

	long cal_strs[DIMS];
	md_calc_strides(DIMS, cal_strs, cal_dims, CFL_SIZE);

	casorati_matrixH(4, kern_dims, cal_dims, cal_strs, back, A_dims, A_backproj);

	// Missing normalization for summed anti-diagonals
	long b = MIN(kern_dims[0], cal_dims[0] - kern_dims[0] + 1); // Minimum of window length and maximum lag

	long norm_dims[DIMS];
	md_singleton_dims(DIMS, norm_dims);

	norm_dims[0] = cal_dims[0];

	complex float* norm = md_alloc(DIMS, norm_dims, CFL_SIZE);

	md_zfill(DIMS, norm_dims, norm, 1. / b);

	for (int i = 0; i < b; i++) {

		norm[i] = 1. / (i + 1);
		norm[cal_dims[0] -1 - i] = 1. / (i + 1);
	}

	long norm_strs[DIMS];
	md_calc_strides(DIMS, norm_strs, norm_dims, CFL_SIZE);

	md_zmul2(DIMS, cal_dims, cal_strs, back, cal_strs, back, norm_strs, norm);

	md_free(norm);
	md_free(A_backproj);
	md_free(PC);
}


extern void ssa_fary(	const long kernel_dims[3],
			const long cal_dims[DIMS],
			const long A_dims[2],
			const complex float* A,
			complex float* U,
			float* S_square,
			complex float* back,
			const int rank,
			const long group)
{
	long N = A_dims[0];
	long M = A_dims[1];

	long AH_dims[2] = { M, N };
	complex float* AH = md_alloc(2, AH_dims, CFL_SIZE);

	md_transpose(2, 0, 1, AH_dims, AH, A_dims, A, CFL_SIZE);
	md_zconj(2, AH_dims, AH, AH);

	long AAH_dims[2] = { N, N };
	complex float* AAH = md_alloc(2, AAH_dims, CFL_SIZE);

	// AAH = A @ AH
	long A2_dims[3] = { N, M, 1 };
	long AH2_dims[3] = { 1, M, N };
	long AAH2_dims[3] = { N, 1, N };

	md_ztenmul(3, AAH2_dims, AAH, A2_dims, A, AH2_dims, AH);


	// AAH = U @ S @ UH
	long U_dims[2] = { N, N };
	complex float* UH = md_alloc(2, U_dims, CFL_SIZE);

	debug_printf(DP_DEBUG3, "SVD of %dx%d matrix...", AAH_dims[0], AAH_dims[1]);

	lapack_svd(N, N, (complex float (*)[N])U, (complex float (*)[N])UH, S_square, (complex float (*)[N])AAH); // NOTE: Lapack destroys AAH!

	debug_printf(DP_DEBUG3, "done\n");

	if (NULL != back) {

		debug_printf(DP_DEBUG3, "Backprojection...\n");

		ssa_backprojection(N, M, kernel_dims, cal_dims, back, A_dims, A, U_dims, U, UH, rank, group);
	}

	md_free(UH);
	md_free(A);
	md_free(AH);
	md_free(AAH);
}


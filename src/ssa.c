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
#include <stdio.h>
#include <math.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/init.h"
#include "num/casorati.h"
#include "num/lapack.h"
#include "num/linalg.h"
#include "num/flpmath.h"

#include "calib/calib.h"
#include "calib/estvar.h"
#include "calib/calmat.h"


static const char usage_str[] = "<src> <EOF> [<S>] [<backprojection>]";
static const char help_str[] =
		"Perform SSA-FARY or Singular Spectrum Analysis. <src>: [samples, coordinates]\n";

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

	for (unsigned int i = 0; i < b; i++) {

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


static void ssa_fary(	const long kernel_dims[3],
			const long cal_dims[DIMS],
			const complex float* cal,
			const char* name_EOF,
			const char* name_S,
			const char* backproj,
			const int rank,
			const long group)
{
	long A_dims[2];
	complex float* A = calibration_matrix(A_dims, kernel_dims, cal_dims, cal);

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
	complex float* U = create_cfl(name_EOF, 2, U_dims);
	complex float* UH = md_alloc(2, U_dims, CFL_SIZE);

	float* S_square = xmalloc(N * sizeof(float));

	debug_printf(DP_DEBUG3, "SVD of %dx%d matrix...", AAH_dims[0], AAH_dims[1]);

	lapack_svd(N, N, (complex float (*)[N])U, (complex float (*)[N])UH, S_square, (complex float (*)[N])AAH); // NOTE: Lapack destroys AAH!

	debug_printf(DP_DEBUG3, "done\n");


	if (NULL != name_S) {

		long S_dims[1] = { N };
		complex float* S = create_cfl(name_S, 1, S_dims);

		for (int i = 0; i < N; i++)
			S[i] = sqrt(S_square[i]) + 0i;

		unmap_cfl(1, S_dims, S);
	}

	if (NULL != backproj) {

		long back_dims[DIMS];
		md_transpose_dims(DIMS, 3, 1, back_dims, cal_dims);

		complex float* back = create_cfl(backproj, DIMS, back_dims);

		debug_printf(DP_DEBUG3, "Backprojection...\n");

		ssa_backprojection(N, M, kernel_dims, cal_dims, back, A_dims, A, U_dims, U, UH, rank, group);
	}

	unmap_cfl(2, U_dims, U);

	md_free(UH);
	md_free(A);
	md_free(AH);
	md_free(AAH);

	xfree(S_square);
}


int main_ssa(int argc, char* argv[])
{
	int window = -1;
	int normalize = 0;
	int rm_mean = 1;
	int rank = 0;
	bool zeropad = true;
	long kernel_dims[3] = { 1, 1, 1 };
	char* name_S = NULL;
	char* backproj = NULL;
	long group = 0;

	const struct opt_s opts[] = {

		OPT_INT('w', &window, "window", "Window length"),
		OPT_CLEAR('z', &zeropad, "Zeropadding [Default: True]"),
		OPT_INT('m', &rm_mean, "0/1", "Remove mean [Default: True]"),
		OPT_INT('n', &normalize, "0/1", "Normalize [Default: False]"),
		OPT_INT('r', &rank, "rank", "Rank for backprojection. r < 0: Throw away first r components. r > 0: Use only first r components."),
		OPT_LONG('g', &group, "bitmask", "Bitmask for Grouping (long value!)"),
	};

	cmdline(&argc, argv, 2, 4, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (-1 == window)
		error("Specify window length '-w'");

	kernel_dims[0] = window;

	char* name_EOF = argv[2];

	if (4 <= argc)
		name_S = argv[3];

	if (5 == argc) {

		backproj = argv[4];

		if (zeropad) {

			debug_printf(DP_INFO, "Zeropadding turned off automatically!\n");

			zeropad = false;
		}

		if ((0 == rank) && (0 == group)
			error("Specify rank or group for backprojection!");

		if (0 == rank)
			assert(0 != group);

		if (0 == group)
			assert(0 != rank);
	}


	long in_dims[DIMS];
	complex float* in = load_cfl(argv[1], DIMS, in_dims);

	if (!md_check_dimensions(DIMS, in_dims, ~(READ_FLAG|PHS1_FLAG)))
		error("Only first two dimensions must be filled!");


	if (rm_mean || normalize) {

		long in_strs[DIMS];
		md_calc_strides(DIMS, in_strs, in_dims, CFL_SIZE);

		long singleton_dims[DIMS];
		long singleton_strs[DIMS];
		md_select_dims(DIMS, ~READ_FLAG, singleton_dims, in_dims);
		md_calc_strides(DIMS, singleton_strs, singleton_dims, CFL_SIZE);

		if (rm_mean) {

			complex float* mean = md_alloc(DIMS, singleton_dims, CFL_SIZE);

			md_zavg(DIMS, in_dims, READ_FLAG, mean, in);
			md_zsub2(DIMS, in_dims, in_strs, in, in_strs, in, singleton_strs, mean);

			md_free(mean);
		}

		if (normalize) {

			complex float* stdv = md_alloc(DIMS, singleton_dims, CFL_SIZE);

			md_zstd(DIMS, in_dims, READ_FLAG, stdv, in);
			md_zdiv2(DIMS, in_dims, in_strs, in, in_strs, in, singleton_strs, stdv);

			md_free(stdv);
		}
	}


	long cal0_dims[DIMS];
	md_copy_dims(DIMS, cal0_dims, in_dims);

	if (zeropad)
		cal0_dims[0] = in_dims[0] - 1 + window;


	complex float* cal = md_alloc(DIMS, cal0_dims, CFL_SIZE);

	md_resize_center(DIMS, cal0_dims, cal, in_dims, in, CFL_SIZE); 

	long cal_dims[DIMS];
	md_transpose_dims(DIMS, 1, 3, cal_dims, cal0_dims);

	debug_printf(DP_INFO, backproj ? "Performing SSA\n" : "Performing SSA-FARY\n");

	// Perform SSA-FARY or SSA
	ssa_fary(kernel_dims, cal_dims, cal, name_EOF, name_S, backproj, rank, group);

	unmap_cfl(DIMS, in_dims, in);
	md_free(cal);

	exit(0);
}



/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <math.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "calib/calmat.h"
#include "calib/ssa.h"


static const char help_str[] =
		"Perform SSA-FARY or Singular Spectrum Analysis. <src>: [samples, coordinates]";


int main_ssa(int argc, char* argv[argc])
{
	const char* src_file = NULL;
	const char* EOF_file = NULL;
	const char* S_file = NULL;
	const char* backproj_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &src_file, "src"),
		ARG_OUTFILE(true, &EOF_file, "EOF"),
		ARG_OUTFILE(false, &S_file, "S"),
		ARG_OUTFILE(false, &backproj_file, "backprojection"),
	};

	int window = -1;
	int normalize = 0;
	int rm_mean = 1;
	int rank = 0;
	bool zeropad = true;
	long kernel_dims[3] = { 1, 1, 1 };
	long group = 0;

	const struct opt_s opts[] = {

		OPT_INT('w', &window, "window", "Window length"),
		OPT_CLEAR('z', &zeropad, "Zeropadding [Default: True]"),
		OPT_INT('m', &rm_mean, "0/1", "Remove mean [Default: True]"),
		OPT_INT('n', &normalize, "0/1", "Normalize [Default: False]"),
		OPT_INT('r', &rank, "rank", "Rank for backprojection. r < 0: Throw away first r components. r > 0: Use only first r components."),
		OPT_LONG('g', &group, "bitmask", "Bitmask for Grouping (long value!)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (-1 == window)
		error("Specify window length '-w'");

	kernel_dims[0] = window;

	if (NULL != backproj_file) {

		if (zeropad) {

			debug_printf(DP_INFO, "Zeropadding turned off automatically!\n");

			zeropad = false;
		}

		if ((0 == rank) && (0 == group))
			error("Specify rank or group for backprojection!");

		if (0 == rank)
			assert(0 != group);

		if (0 == group)
			assert(0 != rank);
	}


	long in_dims[DIMS];
	complex float* in = load_cfl(src_file, DIMS, in_dims);

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


	debug_printf(DP_INFO, backproj_file ? "Performing SSA\n" : "Performing SSA-FARY\n");

	long A_dims[2];
	complex float* A = calibration_matrix(A_dims, kernel_dims, cal_dims, cal);

	long N = A_dims[0];

	long U_dims[2] = { N, N };
	complex float* U = create_cfl(EOF_file, 2, U_dims);

	complex float* back = NULL;

	if (NULL != backproj_file) {

		long back_dims[DIMS];
		md_transpose_dims(DIMS, 3, 1, back_dims, cal_dims);

		back = create_cfl(backproj_file, DIMS, back_dims);
	}

	float* S_square = xmalloc(N * sizeof(float));

	ssa_fary(kernel_dims, cal_dims, A_dims, A, U, S_square, back, rank, group);

	if (NULL != S_file) {

		long S_dims[1] = { N };
		complex float* S = create_cfl(S_file, 1, S_dims);

		for (int i = 0; i < N; i++)
			S[i] = sqrt(S_square[i]) + 0.i;

		unmap_cfl(1, S_dims, S);
	}

	xfree(S_square);

	unmap_cfl(2, U_dims, U);
	unmap_cfl(DIMS, in_dims, in);

	md_free(cal);

	return 0;
}



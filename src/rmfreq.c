/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 *
 * Sebastian Rosenzweig, Nick Scholand, H. Christian M. Holme, Martin Uecker.
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular
 * Spectrum Analysis (SSA-FARY), IEEE Trans. Magn. Imag. (2020), in press.
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"
#include "num/linalg.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"


static const char help_str[] = "Remove angle-dependent frequency";



int main_rmfreq(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* k_file = NULL;
	const char* kcor_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "traj"),
		ARG_INFILE(true, &k_file, "k"),
		ARG_OUTFILE(true, &kcor_file, "k_cor"),
	};

	unsigned int n_harmonics = 5;
	const char* mod_file = NULL;

	const struct opt_s opts[] = {

		OPT_UINT('N', &n_harmonics, "#", "Number of harmonics [Default: 5]"),
		OPT_STRING('M', &mod_file, "file", "Contrast modulation file"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	enum { LAST_DIM = DIMS - 1 };

	// Read k-space
	long k_dims[DIMS];
	complex float* k = load_cfl(k_file, DIMS, k_dims);

	if (md_check_dimensions(DIMS, k_dims, COIL_FLAG|TIME_FLAG|SLICE_FLAG))
		error("Only COIL_DIM, TIME_DIM and SLICE_DIM may have entries!\n");

	// Read trajectory
	long t_dims[DIMS];
	complex float* t = load_cfl(traj_file, DIMS, t_dims);

	if (!md_check_equal_dims(DIMS, t_dims, k_dims, ~(READ_FLAG|PHS1_FLAG|COIL_FLAG)))
		error("k-space and trajectory inconsistent!\n");

	// Modulation file
	long mod_dims[DIMS];
	complex float* mod = NULL;

	if (NULL != mod_file) {

		mod = load_cfl(mod_file, DIMS, mod_dims);

		assert(md_check_equal_dims(DIMS, k_dims, mod_dims, ~0u));
	}


	// Calculate angles from trajectory	
	long angles_dims[DIMS];
	md_select_dims(DIMS, ~(PHS1_FLAG|COIL_FLAG), angles_dims, k_dims);

	complex float* angles = md_alloc(DIMS, angles_dims, CFL_SIZE);

	long t1_dims[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), t1_dims, t_dims);

	complex float* t1 = md_alloc(DIMS, t1_dims, CFL_SIZE);

	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ 0 }, t_dims, t1, t, CFL_SIZE);

	int N = 1;
	for (int i = 0; i < DIMS; i++)
		N = N * angles_dims[i];

	for (int i = 0; i < N; i++) {

		angles[i] = M_PI + atan2f(crealf(t1[3 * i + 0]), crealf(t1[3 * i + 1]));

 		//debug_printf(DP_INFO, "i: %d, angle: %f\n", i, creal(angles[i]) * 360. / 2. / M_PI);
	}

	// negative angles
	complex float* neg_angles = md_alloc(DIMS, angles_dims, CFL_SIZE);

	md_zsmul(DIMS, angles_dims, neg_angles, angles, -1.);

	// Projection matrix
	long n_dims[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, n_dims, k_dims);
	n_dims[LAST_DIM] = 2 * n_harmonics;

	complex float* n = md_alloc(DIMS, n_dims, CFL_SIZE);
	complex float* n_singleton = md_alloc(DIMS, angles_dims, CFL_SIZE);
	complex float* angles1 = md_alloc(DIMS, angles_dims, CFL_SIZE);

	long pos[DIMS] = { 0 };

	int count = 0;

	for (int h = 0; h < (int)n_harmonics; h++) {

		// exp(i * theta * (h+1))
		md_zsmul(DIMS, angles_dims, angles1, angles, (h + 1));
		md_zexpj(DIMS, angles_dims, n_singleton, angles1);

		pos[LAST_DIM] = count;
		md_copy_block(DIMS, pos, n_dims, n, angles_dims, n_singleton, CFL_SIZE);
		count++;

		// exp(- i * theta * (h+1))
		md_zsmul(DIMS, angles_dims, angles1, neg_angles, (h + 1));
		md_zexpj(DIMS, angles_dims, n_singleton, angles1);

		pos[LAST_DIM] = count;

		md_copy_block(DIMS, pos, n_dims, n, angles_dims, n_singleton, CFL_SIZE);
		count++;
	}

	// Projection
	long k_singleton_dims[DIMS];
	md_select_dims(DIMS, TIME_FLAG, k_singleton_dims, k_dims);

	complex float* k_singleton = md_alloc(DIMS, k_singleton_dims, CFL_SIZE);

	long n_part_singleton_dims[DIMS];
	md_select_dims(DIMS, ~SLICE_FLAG, n_part_singleton_dims, n_dims);

	complex float* n_part_singleton = md_alloc(DIMS, n_part_singleton_dims, CFL_SIZE);

	/* Account for contrast change */

	complex float* n_mod = NULL;

	long n_mod_dims[DIMS];
	md_copy_dims(DIMS, n_mod_dims, n_dims);
	n_mod_dims[COIL_DIM] = mod_dims[COIL_DIM];

	long n_mod_strs[DIMS];
	md_calc_strides(DIMS, n_mod_strs, n_mod_dims, CFL_SIZE);

	if (NULL != mod_file) {

		assert(md_check_equal_dims(DIMS, n_dims, mod_dims, ~(COIL_FLAG|(1u << LAST_DIM))));

		long n_strs[DIMS];
		md_calc_strides(DIMS, n_strs, n_dims, CFL_SIZE);

		long mod_strs[DIMS];
		md_calc_strides(DIMS, mod_strs, mod_dims, CFL_SIZE);

		n_mod = md_alloc(DIMS, n_mod_dims, CFL_SIZE);

		md_zmul2(DIMS, n_mod_dims, n_mod_strs, n_mod, n_strs, n, mod_strs, mod);

		unmap_cfl(DIMS, mod_dims, mod);
	}

	long pinv_dims[DIMS];
	md_transpose_dims(DIMS, TIME_DIM, LAST_DIM, pinv_dims, n_part_singleton_dims);

	complex float* pinv = md_alloc(DIMS, pinv_dims, CFL_SIZE);

	long proj_dims[DIMS];

	for (int i = 0; i < DIMS; i++)
		proj_dims[i] = 1;

	proj_dims[0] = 2 * n_harmonics;

	complex float* proj = md_alloc(DIMS, proj_dims, CFL_SIZE);
	complex float* proj_singleton = md_alloc(DIMS, k_singleton_dims, CFL_SIZE);
	complex float* k_cor_singleton = md_alloc(DIMS, k_singleton_dims, CFL_SIZE);

	complex float* k_cor = create_cfl(kcor_file, DIMS, k_dims);

	long pos1[DIMS] = { 0 };

	// Coil-by-coil, Partition-by-Partition correction
	for (int c = 0; c < k_dims[COIL_DIM]; c++) {

		for (int p = 0; p < k_dims[SLICE_DIM]; p++) {

			pos1[SLICE_DIM] = p;
			pos1[COIL_DIM] = 0;

			if (NULL != mod_file) {

				pos1[COIL_DIM] = c;
				md_copy_block(DIMS, pos1, n_part_singleton_dims, n_part_singleton, n_mod_dims, n_mod, CFL_SIZE);

			} else {

				md_copy_block(DIMS, pos1, n_part_singleton_dims, n_part_singleton, n_dims, n, CFL_SIZE);
			}

			pos1[COIL_DIM] = c;
			md_copy_block(DIMS, pos1, k_singleton_dims, k_singleton, k_dims, k, CFL_SIZE);

			// pinv(n)
			mat_pinv_right(2 * n_harmonics, k_dims[TIME_DIM], (complex float (*)[k_dims[TIME_DIM]])pinv, (complex float (*)[2 * n_harmonics])n_part_singleton);

			// k * pinv(n)
			mat_mul(1, k_dims[TIME_DIM], 2 * n_harmonics, (complex float (*)[1])proj, (complex float (*)[1])k_singleton, (complex float (*)[k_dims[TIME_DIM]])pinv);

			// (k * pinv(n)) * n
			mat_mul(1, 2 * n_harmonics, k_dims[TIME_DIM], (complex float (*)[1])proj_singleton, (complex float (*)[1])proj, (complex float (*)[2 * n_harmonics])n_part_singleton);

			// k - (k * pinv(n)) * n
			md_zsub(DIMS, k_singleton_dims, k_cor_singleton, k_singleton, proj_singleton);

			md_copy_block(DIMS, pos1, k_dims, k_cor, k_singleton_dims, k_cor_singleton, CFL_SIZE);
		}
	}

	md_free(angles);
	md_free(angles1);
	md_free(neg_angles);

	md_free(n_singleton);
	md_free(n_part_singleton);
	md_free(n);
	md_free(proj);
	md_free(proj_singleton);
	md_free(pinv);
	md_free(k_cor_singleton);
	md_free(k_singleton);

	unmap_cfl(DIMS, k_dims, k);
	unmap_cfl(DIMS, k_dims, k_cor);

	xfree(n_mod);

	return 0;
}

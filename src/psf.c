/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author: Nick Scholand
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "noncart/nufft.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char help_str[] =
		"Calculate point-spread-function (PSF) of given trajectory.\n";


int main_psf(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* psf_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &traj_file, "trajectory"),
		ARG_OUTFILE(true, &psf_file, "psf"),
	};

	bool oversampled = false;
	bool oversampled_decomp = false;

	const struct opt_s opts[] = {

		OPTL_SET(0, "oversampled", &oversampled, "oversampled psf with linphases in dim 15"),
		OPTL_SET(0, "oversampled-decomposed", &oversampled_decomp, "oversampled psf with linphases in dim 15"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long tdims[DIMS];
	complex float* traj = load_cfl(traj_file, DIMS, tdims);

	if (1 != md_calc_size(DIMS - 3, tdims + 3))
		error("Trajectory has additional dimensions!");

	long img_dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	img_dims[READ_DIM] = tdims[PHS1_DIM];
	img_dims[PHS1_DIM] = tdims[PHS1_DIM];

	complex float* tmp = NULL;

	if (oversampled) {

		for (int i = 0; i < 3; i++)
			if (1 < img_dims[i])
				img_dims[BATCH_DIM] *= 2;

		tmp = compute_psf2(DIMS - 1, img_dims, md_nontriv_dims(3, img_dims), tdims, traj, MD_SINGLETON_DIMS(DIMS), NULL, MD_SINGLETON_DIMS(DIMS), NULL, true, false, false);

	} else if (oversampled_decomp) {

		for (int i = 0; i < 3; i++)
			if (1 < img_dims[i])
				img_dims[BATCH_DIM] *= 2;

		tmp = compute_psf2_decomposed(DIMS - 1, img_dims, md_nontriv_dims(3, img_dims), tdims, traj, MD_SINGLETON_DIMS(DIMS), NULL, MD_SINGLETON_DIMS(DIMS), NULL, true, false, false);

	} else {

		tmp = compute_psf(DIMS, img_dims, tdims, traj, tdims, NULL, NULL, NULL, false, false);
	}


	complex float* psf = create_cfl(psf_file, DIMS, img_dims);

	md_copy(DIMS, img_dims, psf, tmp, CFL_SIZE);

	md_free(tmp);
	unmap_cfl(DIMS, img_dims, psf);

	return 0;
}


/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Moritz Blumenthal
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"
#include "misc/mri.h"



static const char help_str[] = "Compute the Fourier transform of a basis function to be used in the nuFFT.";


int main_nufftbase(int argc, char* argv[argc])
{
	const char* traj_file = NULL;
	const char* out_file = NULL;
	long dims[3];

	struct arg_s args[] = {

		ARG_VEC3(true, &dims, "dimensions"),
		ARG_INFILE(true, &traj_file, "trajectory"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	enum base_type { SINC } base_type = SINC;

	const struct opt_s opts[] = {


	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	long traj_dims[DIMS];	
	complex float* traj = load_cfl(traj_file, DIMS, traj_dims);

	long traj_strs[DIMS];
	md_calc_strides(DIMS, traj_strs, traj_dims, CFL_SIZE);

	long cord_dims[DIMS];
	long cord_strs[DIMS];
	md_select_dims(DIMS, MD_BIT(0), cord_dims, traj_dims);
	md_calc_strides(DIMS, cord_strs, cord_dims, CFL_SIZE);

	switch (base_type) {
	
	case SINC:

		md_zabs(DIMS, traj_dims, traj, traj);
		
		complex float inv_dims[3] = { 1. / dims[0],  1. / dims[1], 1. / dims[2]};
		md_zmul2(DIMS, traj_dims, traj_strs, traj, traj_strs, traj, cord_strs, inv_dims);

		md_zslessequal(DIMS, traj_dims, traj, traj, 0.5);
	}

	long out_dims[DIMS];
	long out_strs[DIMS];
	md_select_dims(DIMS, ~MD_BIT(0), out_dims, traj_dims);
	md_calc_strides(DIMS, out_strs, out_dims, CFL_SIZE);

	complex float* odata = create_cfl(out_file, DIMS, out_dims);
	md_zfill(DIMS, out_dims, odata, 1.);

	for (int i = 0; i < traj_dims[0]; i++)
		md_zmul2(DIMS, out_dims, out_strs, odata, out_strs, odata, traj_strs, traj + i);

	unmap_cfl(DIMS, traj_dims, traj);
	unmap_cfl(DIMS, out_dims, odata);

	return 0;
}

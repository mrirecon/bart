/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
 * References:
 *
 * Parzen E. On the estimation of a probability density
 * function and the mode. Annals of Mathematical Statistics
 * 1962;33:1065-1076.
 *
 * Mattes D, Haynor DR, Vesselle H, Lewellen TK, Eubank W.
 * PET-CT image registration in the chest using free-form deformations.
 * IEEE TMI 2023;22:120-8.
 */

#include <assert.h>
#include <complex.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "nlops/nlop.h"

#include "motion/affine.h"

#ifndef DIMS
#define DIMS 16
#endif

enum AFFINE_TYPE { AFFINE_TRANS, AFFINE_RIGID, AFFINE_ALL };

static const char help_str[] = "Affine registration of reference of <input> and  <moved>.";

int main_affinereg(int argc, char* argv[argc])
{
	const char* ref_file = NULL;
	const char* affine_file = NULL;
	const char* mov_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ref_file, "reference"),
		ARG_INFILE(true, &mov_file, "moved"),
		ARG_OUTFILE(true, &affine_file, "affine"),
	};

	enum AFFINE_TYPE aff = AFFINE_RIGID;

	float factors[5] = { 1., 0.5, 0.25, 0.125, 0.0625 };
	float sigmas[5] = { 0., 2., 4., 8., 16. };

	const char* msk_mov_file = NULL;
	const char* msk_ref_file = NULL;

	const struct opt_s opts[] = {

		OPT_SET('g', &bart_use_gpu, "use gpu (if available)"),

		OPTL_INFILE(0, "mask-reference", &msk_ref_file, "file", "binary mask for the reference image"),
		OPTL_INFILE(0, "mask-moved", &msk_mov_file, "file", "binary mask for the moved image"),

		OPT_SELECT('T', enum AFFINE_TYPE, &aff, AFFINE_TRANS, "Translation"),
		OPT_SELECT('R', enum AFFINE_TYPE, &aff, AFFINE_RIGID, "Rigid transformation (default)"),
		OPT_SELECT('A', enum AFFINE_TYPE, &aff, AFFINE_ALL, "All degrees of freedom"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init_gpu_support();

	long rdims[DIMS];
	long mdims[DIMS];

	complex float* ref_ptr = load_cfl(ref_file, DIMS, rdims);
	complex float* mov_ptr = load_cfl(mov_file, DIMS, mdims);

	if (   (0 != (~7ul & md_nontriv_dims(DIMS, rdims)))
	    || (0 != (~7ul & md_nontriv_dims(DIMS, mdims))))
			error("Affine registration only supports the first three dimensions.\nUse bart looping for higher dimensions.\n");

	md_zabs(DIMS, mdims, mov_ptr, mov_ptr);
	md_zabs(DIMS, rdims, ref_ptr, ref_ptr);

	if ((NULL == msk_mov_file) != (NULL == msk_ref_file))
		error("Need both masks or none.\n");


	complex float* msk_ref_ptr = NULL;
	complex float* msk_mov_ptr = NULL;
	
	if (NULL != msk_mov_file) {

		long tdims[DIMS];

		msk_mov_ptr = load_cfl(msk_mov_file, DIMS, tdims);
		assert(md_check_equal_dims(DIMS, mdims, tdims, ~0ul));

		msk_ref_ptr = load_cfl(msk_ref_file, DIMS, tdims);
		assert(md_check_equal_dims(DIMS, rdims, tdims, ~0ul));
	} 

	const struct nlop_s* trafo = NULL; // false positive

	switch (aff) {

	case AFFINE_ALL:
		trafo = (1 == rdims[2]) ? nlop_affine_2D() : nlop_affine_3D();
		break;

	case AFFINE_RIGID:
		trafo = (1 == rdims[2]) ? nlop_affine_rigid_2D() : nlop_affine_rigid_3D();
		break;

	case AFFINE_TRANS:
		trafo = (1 == rdims[2]) ? nlop_affine_translation_2D() : nlop_affine_translation_3D();
		break;
	}

	long aff_dims[DIMS] = { 3, 4, [ 2 ... DIMS - 1 ] = 1 };
	complex float* affine = create_cfl(affine_file, DIMS, aff_dims);

	affine_init_id(affine);

	affine_reg(bart_use_gpu, false, affine, trafo, rdims, ref_ptr, msk_ref_ptr, mdims, mov_ptr, msk_mov_ptr, 3, sigmas, factors);
	nlop_free(trafo);

	affine_debug(DP_INFO, affine);

	unmap_cfl(DIMS, mdims, mov_ptr);
	unmap_cfl(DIMS, rdims, ref_ptr);
	unmap_cfl(DIMS, aff_dims, affine);	
	unmap_cfl(DIMS, mdims, msk_mov_ptr);
	unmap_cfl(DIMS, rdims, msk_ref_ptr);

	return 0;
}

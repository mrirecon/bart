/* Copyright 2024-2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 *
 *
 * Kim D, Cauley SF, Nayak KS, Leahy RM, Haldar JP.
 * Region-optimized virtual (ROVir) coils: Localization and/or suppression
 * of spatial regions using sensor-domain beamforming
 * Magn Reson Med 2021; 86:197-212.
 */


#include <complex.h>
#include <stdbool.h>

#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/lapack.h"
#include "num/linalg.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/opts.h"

#define DIMS 16



static const char help_str[] = "Compute coil compression matrix using ROVir.";


int main_rovir(int argc, char* argv[argc])
{
	const char* pos_file = NULL;
	const char* neg_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &pos_file, "positive signal"),
		ARG_INFILE(true, &neg_file, "negative negative"),
		ARG_OUTFILE(true, &out_file, "orthogonal transform"),
	};

	const struct opt_s opts[] = {	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	long pos_dims[DIMS];
	long neg_dims[DIMS];

	complex float* pos = load_cfl(pos_file, DIMS, pos_dims);
	complex float* neg = load_cfl(neg_file, DIMS, neg_dims);

	assert(md_check_equal_dims(DIMS, pos_dims, neg_dims, ~0UL));

	long out_dims[DIMS];
	md_select_dims(DIMS, COIL_FLAG, out_dims, pos_dims);
	out_dims[MAPS_DIM] = out_dims[COIL_DIM];

	long tra_dims[DIMS];
	md_transpose_dims(DIMS, MAPS_DIM, COIL_DIM, tra_dims, pos_dims);

	complex float* A = md_alloc(DIMS, out_dims, CFL_SIZE);
	complex float* B = md_alloc(DIMS, out_dims, CFL_SIZE);

	complex float* out = create_cfl(out_file, DIMS, out_dims);

	md_ztenmulc(DIMS, out_dims, A, pos_dims, pos, tra_dims, pos);
	md_ztenmulc(DIMS, out_dims, B, pos_dims, neg, tra_dims, neg);

	unmap_cfl(DIMS, pos_dims, pos);
	unmap_cfl(DIMS, neg_dims, neg);

	long N = out_dims[COIL_DIM];
	float eigen[N];

	lapack_geig(N, eigen,
		    MD_CAST_ARRAY2(complex float, DIMS, out_dims, A, COIL_DIM, MAPS_DIM),
		    MD_CAST_ARRAY2(complex float, DIMS, out_dims, B, COIL_DIM, MAPS_DIM));

	float vals[N];
	gram_schmidt(N, N, vals, MD_CAST_ARRAY2(complex float, DIMS, out_dims, A, COIL_DIM, MAPS_DIM));

	md_flip(DIMS, out_dims, MAPS_FLAG, out, A, CFL_SIZE);

	md_free(A);
	md_free(B);

	unmap_cfl(DIMS, out_dims, out);

	return 0;
}


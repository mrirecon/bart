/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * References:
 *
 * Peter J. Shin, Peder E.Z. Larson, Michael A. Ohliger, Michael Elad,
 * John M. Pauly, Daniel B. Vigneron and Michael Lustig, Calibrationless
 * Parallel Imaging Reconstruction Based on Structured Low-Rank Matrix 
 * Completion, 2013, accepted to Magn Reson Med.
 *
 * Zhongyuan Bi, Martin Uecker, Dengrong Jiang, Michael Lustig, and Kui Ying.
 * Robust Low-rank Matrix Completion for sparse motion correction in auto 
 * calibration PI. Annual Meeting ISMRM, Salt Lake City 2013, 
 * In Proc. Intl. Soc. Mag. Recon. Med 21; 2584 (2013)
 */

#include <math.h>
#include <complex.h>

#include "num/init.h"
#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"

#include "sake/sake.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char help_str[] =
		"Use SAKE algorithm to recover a full k-space from undersampled\n"
		"data using low-rank matrix completion.";

int main_sake(int argc, char* argv[argc])
{
	const char* ksp_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	float alpha = 0.22;
	int iter = 50;
	float lambda = 1.;

	const struct opt_s opts[] = {

		OPT_INT('i', &iter, "iter", "number of iterations"),
		OPT_FLOAT('s', &alpha, "size", "rel. size of the signal subspace"),
		OPT_FLOAT('o', &lambda, "", "()"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	assert((0. <= alpha) && (alpha <= 1.));
	assert(iter >= 0);
	assert((0. <= lambda) && (lambda <= 1.));

	long dims[DIMS];

	num_init();
	
	complex float* in_data = load_cfl(ksp_file, DIMS, dims);
	complex float* out_data = create_cfl(out_file, DIMS, dims);

	lrmc(alpha, iter, lambda, DIMS, dims, out_data, in_data);

	unmap_cfl(DIMS, dims, out_data);
	unmap_cfl(DIMS, dims, in_data);

	return 0;
}



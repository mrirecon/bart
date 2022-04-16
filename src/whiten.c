/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2017	Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"
#include "num/lapack.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char help_str[] = "Apply multi-channel noise pre-whitening on <input> using noise data <ndata>.\n"
				"Optionally output whitening matrix and noise covariance matrix";


static void whiten(const long dims[DIMS], complex float* out, const long mat_dims[DIMS], const complex float* mat, const complex float* in)
{
	long trp_dims[DIMS];

	md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, trp_dims, dims);
	md_zmatmul(DIMS, trp_dims, out, mat_dims, mat, dims, in);
}

/* 
 * Calculate noise covariance matrix. Assumes noise is zero-mean
 */
static void calc_covar(const long mat_dims[DIMS], complex float* covar, const long noise_dims[DIMS], const complex float* ndata)
{
	long trp_dims[DIMS];
	md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, trp_dims, noise_dims);

	md_zmatmulc(DIMS, mat_dims, covar, trp_dims, ndata, noise_dims, ndata);
	md_zsmul(DIMS, mat_dims, covar, covar, 1. / (noise_dims[READ_DIM] - 1));
}


/* 
 * Calculate noise whitening matrix W = inv(L), where N = L * L^H is the Cholesky decomposition of noise N
 */
static void calc_optmat(const long mat_dims[DIMS], complex float* optmat, const complex float* covar)
{
	long N = mat_dims[COIL_DIM];

	complex float* chol = md_alloc(DIMS, mat_dims, CFL_SIZE);

	md_copy(DIMS, mat_dims, chol, covar, CFL_SIZE);

	lapack_cholesky_lower(N, MD_CAST_ARRAY2(complex float, DIMS, mat_dims, chol, COIL_DIM, MAPS_DIM));
	lapack_trimat_inverse_lower(N, MD_CAST_ARRAY2(complex float, DIMS, mat_dims, chol, COIL_DIM, MAPS_DIM));

	for (int i = 0; i < N; i++)
		for (int j = 0; j < i; j++)
			chol[i * N + j] = 0.;

	md_transpose(DIMS, COIL_DIM, MAPS_DIM, mat_dims, optmat, mat_dims, chol, CFL_SIZE);
	
	md_free(chol);
}


int main_whiten(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* ndata_file = NULL;
	const char* out_file = NULL;
	const char* optmat_ofile = NULL;
	const char* covar_ofile = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &in_file, "input"),
		ARG_INFILE(true, &ndata_file, "ndata"),
		ARG_OUTFILE(true, &out_file, "output"),
		ARG_OUTFILE(false, &optmat_ofile, "optmat_out"),
		ARG_OUTFILE(false, &covar_ofile, "covar_out"),
	};


	const char* optmat_ifile = NULL;
	const char* covar_ifile = NULL;
	bool normalize = false;

	const struct opt_s opts[] = {

		OPT_INFILE('o', &optmat_ifile, "<optmat_in>", "use external whitening matrix <optmat_in>"),
		OPT_INFILE('c', &covar_ifile, "<covar_in>", "use external noise covariance matrix <covar_in>"),
		OPT_SET('n', &normalize, "normalize variance to 1 using noise data <ndata>"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	long noise_dims[DIMS];
	long mat_dims[DIMS];

	complex float* idata = load_cfl(in_file, DIMS, dims);
	complex float* ndata = load_cfl(ndata_file, DIMS, noise_dims);
	complex float* odata = create_cfl(out_file, DIMS, dims);

	md_select_dims(DIMS, COIL_FLAG, mat_dims, noise_dims);
	mat_dims[MAPS_DIM] = mat_dims[COIL_DIM];

	complex float* optmat_in = NULL;
	complex float* optmat_out = NULL;

	complex float* covar_in = NULL;
	complex float* covar_out = NULL;


	if (NULL != optmat_ofile)
		optmat_out = create_cfl(optmat_ofile, DIMS, mat_dims);
	else
		optmat_out = anon_cfl(NULL, DIMS, mat_dims);

	if (NULL != covar_ofile)
		covar_out = create_cfl(covar_ofile, DIMS, mat_dims);
	else
		covar_out = anon_cfl(NULL, DIMS, mat_dims);

	if (NULL != covar_ifile) {

		covar_in = load_cfl(covar_ifile, DIMS, mat_dims);

		md_copy(DIMS, mat_dims, covar_out, covar_in, CFL_SIZE);

		unmap_cfl(DIMS, mat_dims, covar_in);

	} else {

		calc_covar(mat_dims, covar_out, noise_dims, ndata);
	}


	if (NULL != optmat_ifile) {

		optmat_in = load_cfl(optmat_ifile, DIMS, mat_dims);

		md_copy(DIMS, mat_dims, optmat_out, optmat_in, CFL_SIZE);

		unmap_cfl(DIMS, mat_dims, optmat_in);

	} else {

		calc_optmat(mat_dims, optmat_out, covar_out);
	}


	whiten(dims, odata, mat_dims, optmat_out, idata);


	if (normalize) {

		long std_dims[DIMS];
		md_singleton_dims(DIMS, std_dims);

		complex float* nwhite = md_alloc(DIMS, noise_dims, CFL_SIZE);
		complex float* nstdev = md_alloc(DIMS, std_dims, CFL_SIZE);

		// get scale factor by whitening the noise data and taking stdev
		whiten(noise_dims, nwhite, mat_dims, optmat_out, ndata);

		md_zstd(DIMS, noise_dims, ~0, nstdev, nwhite);

		float stdev = md_zasum(DIMS, std_dims, nstdev);

		md_zsmul(DIMS, dims, odata, odata, 1. / stdev);

		debug_printf(DP_DEBUG1, "standard deviation scaling: %.6e\n", stdev);

		md_free(nwhite);
		md_free(nstdev);
	}

	unmap_cfl(DIMS, dims, idata);
	unmap_cfl(DIMS, noise_dims, ndata);
	unmap_cfl(DIMS, dims, odata);
	unmap_cfl(DIMS, mat_dims, optmat_out);
	unmap_cfl(DIMS, mat_dims, covar_out);

	return 0;
}

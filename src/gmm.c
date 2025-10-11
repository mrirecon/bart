/* Copyright 2025. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "num/linalg.h"
#include "num/init.h"
#include "num/gaussians.h"
#include "num/loop.h"

#include "misc/debug.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char help_str[] = "Evaluate Gaussian mixture.";



int main_gmm(int argc, char* argv[argc])
{
	const char* mean_file = NULL;
	const char* var_file = NULL;
	const char* wght_file = NULL;
	const char* pnt_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &wght_file, "weights"),
		ARG_INFILE(true, &mean_file, "mean values"),
		ARG_INFILE(true, &var_file, "variances"),
		ARG_INFILE(false, &pnt_file, "evaluate at these points"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool score = false;
	bool sample = false;

	const struct opt_s opts[] = {

		OPTL_SET('\0', "score", &score, "compute score"),
		OPTL_SET('\0', "sample", &sample, "create sample"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (sample) {

		if (score)
			error("invalid options");
		
		if (NULL != pnt_file)
			error("no input when sampling");
	} else {

		if (NULL == pnt_file)
			error("input is missing");
	}

	num_init();

	enum { D = 4 };

	/* 0: length
	 * 1: nr of Gaussians
	 **/

	long dims[D];
	const complex float* mean = load_cfl(mean_file, D, dims);

	int N = dims[0];
	int M = dims[1];

	assert(!md_check_dimensions(D, dims, MD_BIT(0) | MD_BIT(1)));

	debug_printf(DP_DEBUG1, "Gaussians: %d Dimension: %d\n", M, N);

	long vdims[D];
	const complex float* vars = load_cfl(var_file, D, vdims);

	assert(md_check_compat(D, MD_BIT(0), dims, vdims));
	assert(!md_check_dimensions(D, vdims, ~MD_BIT(0)));

	long wdims[D];
	const complex float* wght = load_cfl(wght_file, D, wdims);

	assert(md_check_compat(D, MD_BIT(0), dims, wdims));
	assert(!md_check_dimensions(D, wdims, ~MD_BIT(0)));

	float *wght2 = md_alloc(D, wdims, FL_SIZE);

	for (int m = 0; m < M; m++) 
		wght2[m] = crealf(wght[m]);

	unmap_cfl(D, wdims, wght);


	long vdims1[3] = { 1, 1, M };

	long vdims2[3] = { N, N, M };
	complex float *vars2 = md_alloc(3, vdims2, CFL_SIZE);

	long edims[3] = { N, N, 1 };
	complex float *eye = md_alloc(3, edims, CFL_SIZE);

	mat_identity(N, N, (complex float(*)[])eye);

	md_ztenmul(3, vdims2, vars2, edims, eye, vdims1, vars);
	md_free(eye);

	unmap_cfl(D, vdims, vars);

	long odims[4] = { 1, 1, 1, 1 };

	if (score || sample)
		odims[0] = N;

	complex float *out = create_cfl(out_file, 4, odims);

	if (sample) {

		gaussian_mix_sample(M, N, wght2, (void*)mean, (void*)vars2, out);

	} else {

		long pdims[4];
		const complex float* p = load_cfl(pnt_file, 4, pdims);

		assert(md_check_compat(D, MD_BIT(1), dims, pdims));
		assert(!md_check_dimensions(D, pdims, ~MD_BIT(1)));

		if (score) {

			gaussian_mix_score(M, N, wght2, (void*)mean, (void*)vars2, p, out);

		} else {

			*out = gaussian_mix_pdf(M, N, wght2, (void*)mean, (void*)vars2, p);
		}

		unmap_cfl(4, pdims, p);
	}

	md_free(vars2);
	md_free(wght2);

	unmap_cfl(D, dims, mean);

	return 0;
}


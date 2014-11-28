/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2014 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/casorati.h"
#include "num/lapack.h"
#include "num/la.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "calmat.h"


static complex float* calibration_matrix_priv(long calmat_dims[2], const long kdims[3], const long calreg_dims[4], const complex float* data)
{
	long kernel_dims[4];
	md_copy_dims(3, kernel_dims, kdims);
	kernel_dims[3] = calreg_dims[3];

	casorati_dims(4, calmat_dims, kernel_dims, calreg_dims);

	complex float* cm = md_alloc_sameplace(2, calmat_dims, CFL_SIZE, data);

	long calreg_strs[4];
	md_calc_strides(4, calreg_strs, calreg_dims, CFL_SIZE);
	casorati_matrix(4, kernel_dims, calmat_dims, cm, calreg_dims, calreg_strs, data);

	return cm;
}


static complex float* pattern_matrix(long pcm_dims[2], const long kdims[3], const long calreg_dims[4], const complex float* data)
{
	// estimate pattern
	long pat_dims[4];
	md_select_dims(4, ~(1 << COIL_DIM), pat_dims, calreg_dims);
	complex float* pattern = md_alloc_sameplace(4, pat_dims, CFL_SIZE, data);
	estimate_pattern(4, calreg_dims, COIL_DIM, pattern, data);

	// compute calibration matrix of pattern
	complex float* pm = calibration_matrix_priv(pcm_dims, kdims, pat_dims, pattern);
	md_free(pattern);

	return pm;
}

complex float* calibration_matrix(long calmat_dims[2], const long kdims[3], const long calreg_dims[4], const complex float* data)
{
	return calibration_matrix_priv(calmat_dims, kdims, calreg_dims, data);
}


complex float* calibration_matrix2(long calmat_dims[2], const long kdims[3], const complex float* mask, const long calreg_dims[4], const complex float* data)
{
	assert(NULL == mask);

	long pcm_dims[2];
	complex float* pm = pattern_matrix(pcm_dims, kdims, calreg_dims, data);

	long pcm_strs[2];
	md_calc_strides(2, pcm_strs, pcm_dims, CFL_SIZE);

	// number of samples for each patch
	long msk_dims[2];
	md_select_dims(2, ~(1 << 1), msk_dims, pcm_dims);

	long msk_strs[2];
	md_calc_strides(2, msk_strs, msk_dims, CFL_SIZE);

	complex float* msk = md_alloc(2, msk_dims, CFL_SIZE);
	md_clear(2, msk_dims, msk, CFL_SIZE);
	md_zfmacc2(2, pcm_dims, msk_strs, msk, pcm_strs, pm, pcm_strs, pm);
	md_free(pm);

	// fully sampled?
	md_zcmp2(2, msk_dims, msk_strs, msk, msk_strs, msk,
			(long[2]){ 0, 0 }, &(complex float){ pcm_dims[1] });

	debug_printf(DP_DEBUG1, "%ld/%ld fully-sampled patches.\n",
				(long)pow(md_znorm(2, msk_dims, msk), 2.), pcm_dims[0]);

	complex float* tmp = calibration_matrix_priv(calmat_dims, kdims, calreg_dims, data);

	// mask out incompletely sampled patches...
	long calmat_strs[2];
	md_calc_strides(2, calmat_strs, calmat_dims, CFL_SIZE);
	md_zmul2(2, calmat_dims, calmat_strs, tmp, calmat_strs, tmp, msk_strs, msk);

	return tmp;
}



void covariance_function(const long kdims[3], unsigned int N, complex float cov[N][N], const long calreg_dims[4], const complex float* data)
{
	long calmat_dims[2];
	complex float* cm = calibration_matrix(calmat_dims, kdims, calreg_dims, data);

	unsigned int L = calmat_dims[0];
	assert(N == calmat_dims[1]);

	gram_matrix(N, cov, L, MD_CAST_ARRAY2(const complex float, 2, calmat_dims, cm, 0, 1));

	md_free(cm);
}



void calmat_svd(const long kdims[3], unsigned int N, complex float cov[N][N], float* S, const long calreg_dims[4], const complex float* data)
{
	long calmat_dims[2];
	complex float* cm = calibration_matrix(calmat_dims, kdims, calreg_dims, data);

	unsigned int L = calmat_dims[0];
	assert(N == calmat_dims[1]);

	complex float (*U)[L] = xmalloc(L * L * CFL_SIZE);

	svd_econ(L, N, U, cov, S, MD_CAST_ARRAY2(const complex float, 2, calmat_dims, cm, 0, 1));

	free(U);
	md_free(cm);
}


/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 *
 *
 * Peter J. Shin, Peder E.Z. Larson, Michael A. Ohliger, Michael Elad,
 * John M. Pauly, Daniel B. Vigneron and Michael Lustig, Calibrationless
 * Parallel Imaging Reconstruction Based on Structured Low-Rank Matrix 
 * Completion, Magn Reson Med. Epub (2014)
 *
 * Zhongyuan Bi, Martin Uecker, Dengrong Jiang, Michael Lustig, and Kui Ying.
 * Robust Low-rank Matrix Completion for sparse motion correction in auto 
 * calibration PI. Annual Meeting ISMRM, Salt Lake City 2013, 
 * In Proc. Intl. Soc. Mag. Recon. Med 21; 2584 (2013)
 */

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <complex.h>

#include "num/lapack.h"
#include "num/linalg.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/casorati.h"

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"

#include "sake.h"

#undef DIMS // FIXME
#define DIMS 16

#if 0
static float thresh(float lambda, float x)
{
	float norm = fabs(x);
	float red = norm - lambda;
	return (red > 0.) ? ((red / norm) * (x)) : 0.;
}
#endif

static complex float cthresh(float lambda, complex float x)
{
	float norm = cabsf(x);
	float red = norm - lambda;
	return (red > 0.) ? ((red / norm) * x) : 0.;
}



static void robust_consistency(float lambda, const long dims[5], complex float* dst, const complex float* pattern, const complex float* kspace)
{
	assert(1 == dims[4]);

	size_t size = md_calc_size(5, dims);

	for (unsigned int i = 0; i < size; i++)
		if (1. == pattern[i % (size / dims[3])])
			dst[i] = kspace[i] + cthresh(lambda, dst[i] - kspace[i]);
}

#if 1
#define RAVINE
#endif
#ifdef RAVINE
static void ravine(unsigned int N, const long dims[N], float* ftp, complex float* xa, complex float* xb)
{
        float ft = *ftp;
        float tfo = ft;

        ft = (1.f + sqrtf(1.f + 4.f * ft * ft)) / 2.f;
        *ftp = ft;

	md_swap(N, dims, xa, xb, CFL_SIZE);
	complex float val = (1.f - tfo) / ft - 1.f;

	long dims1[N];
	md_singleton_dims(N, dims1);

	long strs1[N];
	long strs[N];
	md_calc_strides(N, strs1, dims1, CFL_SIZE);
	md_calc_strides(N, strs, dims, CFL_SIZE);

	md_zfmac2(N, dims, strs, xa, strs1, &val, strs, xa);
	val *= -1.;
        md_zfmac2(N, dims, strs, xa, strs1, &val, strs, xb);
}
#endif





static void lowrank(float alpha, int D, const long dims[D], complex float* matrix)
{
	assert(1 == dims[MAPS_DIM]);

	debug_printf(DP_DEBUG3, "mat_dims = \t");
	debug_print_dims(DP_DEBUG3, D, dims);

	long kern_min[4] = { 6, 6, 6, dims[COIL_DIM] };
	long kern_dims[D];

	md_set_dims(D, kern_dims, 1);
	md_min_dims(4, ~0u, kern_dims, kern_min, dims);

	debug_printf(DP_DEBUG3, "kern_dims = \t");
	debug_print_dims(DP_DEBUG3, D, kern_dims);

	long calmat_dims[2];
	casorati_dims(D, calmat_dims, kern_dims, dims);

	debug_printf(DP_DEBUG3, "calmat_dims = \t");
	debug_print_dims(DP_DEBUG3, 2, calmat_dims);

	complex float* calmat = md_alloc(2, calmat_dims, CFL_SIZE);

	long str[D];
	md_calc_strides(D, str, dims, CFL_SIZE);

	casorati_matrix(D, kern_dims, calmat_dims, calmat, dims, str, matrix);

	int N = calmat_dims[0];
	int M = calmat_dims[1];

	debug_printf(DP_INFO, "%dx%d\n", N, M);


	if (-1. != alpha) {

		long dimsU[2] = { N, N };
		long dimsV[2] = { M, M };

		complex float* U = md_alloc(2, dimsU, CFL_SIZE);
		complex float* VT = md_alloc(2, dimsV, CFL_SIZE);
		//	complex float* U = create_cfl("U", 2, dimsU);
		//	complex float* VT = create_cfl("VT", 2, dimsV);
		float* S = xmalloc(MIN(N, M) * sizeof(float));

		debug_printf(DP_INFO, "SVD..\n");

		//lapack_svd(N, M, (complex float (*)[M])U, (complex float (*)[N])VT, S, (complex float (*)[N])calmat);
		lapack_svd_econ(N, M, (complex float (*)[M])U, (complex float (*)[N])VT, S, (complex float (*)[N])calmat); // CHECK

		debug_printf(DP_INFO, "done.\n");

		// put it back together
		long dimU2[2] = { N, MIN(N, M) };
		long dimV2[2] = { MIN(N, M), M };
		complex float* U2 = md_alloc(2, dimU2, CFL_SIZE);
		complex float* V2 = md_alloc(2, dimV2, CFL_SIZE);
		md_resize(2, dimU2, U2, dimsU, U, CFL_SIZE);
		md_resize(2, dimV2, V2, dimsV, VT, CFL_SIZE);

		for (int i = 0; i < M; i++) {

			//		printf("%f\t", S[i]);

			for (int j = 0; j < MIN(N, M); j++)
				//V2[i * MIN(N, M) + j] *= thresh(alpha, S[j]);
				V2[i * MIN(N, M) + j] *= (j < alpha * (float)MIN(N, M)) ? S[j] : 0.; //  thresh(alpha, S[j]);
		}

		mat_mul(M, MIN(M, N), N, (complex float (*)[N])calmat,
				(const complex float (*)[MIN(M, N)])V2, (const complex float (*)[N])U2);

		md_free(U);
		md_free(U2);
		md_free(VT);
		md_free(V2);
		free(S);
	}

	md_clear(D, dims, matrix, CFL_SIZE);
	casorati_matrixH(D, kern_dims, dims, str, matrix, calmat_dims, calmat);
	md_zsmul(D, dims, matrix, matrix, 1. / (double)md_calc_size(3, kern_dims)); // FIXME: not right at the border

	md_free(calmat);
}



void lrmc(float alpha, int iter, float lambda, int N, const long dims[N], complex float* out, const complex float* in)
{
	long dims1[N];
	md_select_dims(N, ~COIL_FLAG, dims1, dims);

	md_copy(N, dims, out, in, CFL_SIZE);

	complex float* pattern = md_alloc(N, dims1, CFL_SIZE);

	//assert(5 == N);
	estimate_pattern(N, dims, COIL_FLAG, pattern, in);

	complex float* comp = md_alloc(N, dims1, CFL_SIZE);
	md_zfill(N, dims1, comp, 1.);

	lowrank(-1., N, dims1, comp);

#ifdef RAVINE
	complex float* o = md_alloc(N, dims, CFL_SIZE);
	md_clear(N, dims, o, CFL_SIZE);
	float fl = 1.;
#endif

	long strs1[N];
	md_calc_strides(N, strs1, dims1, CFL_SIZE);

	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	

	for (int i = 0; i < iter; i++) {

		debug_printf(DP_INFO, "%d\n", i);

		if (-1. != lambda)
			robust_consistency(lambda, dims, out, pattern, in);
		else
			data_consistency(dims, out, pattern, in, out);

		lowrank(alpha, N, dims, out);
		md_zdiv2(N, dims, strs, out, strs, out, strs1, comp);
#ifdef RAVINE
		ravine(N, dims, &fl, out, o);
#endif	
	}
	
	debug_printf(DP_INFO, "Done.\n");
#ifdef RAVINE
	md_free(o);
#endif
	md_free(comp);
	md_free(pattern);
}



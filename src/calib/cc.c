/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2014	Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013		Dara Bahri <dbahri123@gmail.com>
 *
 *
 * Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR. A software channel
 * compression technique for faster reconstruction with many channels.
 * Magn Reson Imaging 2008; 26:133-141.
 *
 * Buehrer M, Pruessmann KP, Boesiger P, Kozerke S. Array compression for MRI
 * with large coil arrays. Magn Reson Med 2007, 57: 1131–1139.
 *
 * Zhang T, Pauly JM, Vasanawala SS, Lustig M. Coil compression for
 * accelerated imaging with cartesian sampling. Magn Reson Med 2013;
 * 69:571-582.
 *
 * Bahri D, Uecker M, Lustig M. ESPIRiT-Based Coil Compression for
 * Cartesian Sampling,  Annual Meeting ISMRM, Salt Lake City 2013,
 * In: Proc Intl Soc Mag Reson Med 21:2657
 */
 
#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/lapack.h"
#include "num/linalg.h"

#include "misc/debug.h"
#include "misc/mri.h"

#include "calib/calib.h"

#include "cc.h"


void scc(const long out_dims[DIMS], complex float* out_data, const long caldims[DIMS], const complex float* cal_data)
{
	int channels = caldims[COIL_DIM];

	assert(1 == md_calc_size(3, out_dims));
	assert(out_dims[COIL_DIM] == channels);
	assert(out_dims[MAPS_DIM] == channels);

	complex float tmp[channels][channels];


	size_t csize = md_calc_size(3, caldims);
	gram_matrix(channels, tmp, csize, (const complex float (*)[csize])cal_data);

	float vals[channels];
	lapack_eig(channels, vals, tmp);

	md_flip(DIMS, out_dims, MAPS_FLAG, out_data, tmp, CFL_SIZE);


	debug_printf(DP_DEBUG1, "Energy:");

	float sum = 0.;
	for (int i = 0; i < channels; i++)
		sum += vals[i];

	for (int i = 0; i < channels; i++)
		debug_printf(DP_DEBUG1, " %.3f", vals[channels - 1 - i] / sum);

	debug_printf(DP_DEBUG1, "\n");
}



static void align1(int M, int N, complex float out[M][N], const complex float in1[M][N], const complex float in2[M][N])
{
	assert(M <= N);
#if 1
	complex float in1T[N][M];
	complex float C[M][M];
	complex float U[M][M];
	complex float VH[M][M];
	float S[M];
	complex float P[M][M];

	mat_adjoint(M, N, in1T, in1);	// A_{x-1}^H
	mat_mul(M, N, M, C, in2, in1T);	// C = A_{x} A_{x-1}^H
	// VH and U are switched here because SVD uses column-major arrays
	lapack_svd(M, M, VH, U, S, C);		// U S V^H = C
	mat_mul(M, M, M, C, U, VH);	// U V^H
	mat_adjoint(M, M, P, C);	// P_x = V U^H
	mat_mul(M, M, N, out, P, in2);	// A_{x} <- P_x A_{x}
#else
	mat_copy(M, N, out, in2);
#endif
}



static void align_ro2(const long dims[DIMS], int start, int end, complex float* odata, const complex float* idata)
{
	int dir = (start < end) ? 1 : -1;

	long tmp_dims[DIMS];
	md_select_dims(DIMS, ~READ_FLAG, tmp_dims, dims);

	complex float* tmp1 = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	complex float* tmp2 = md_alloc(DIMS, tmp_dims, CFL_SIZE);
	complex float* tmp3 = md_alloc(DIMS, tmp_dims, CFL_SIZE);

	md_copy_block(DIMS, (long[DIMS]){ [READ_DIM] = start }, tmp_dims, tmp1, dims, idata, CFL_SIZE);

	if (dir)
		md_copy_block(DIMS, (long[DIMS]){ [READ_DIM] = start }, dims, odata, tmp_dims, tmp1, CFL_SIZE);

	for (int i = start; i != end - dir; i += dir) {

		md_copy_block(DIMS, (long[DIMS]){ [READ_DIM] = i + dir }, tmp_dims, tmp2, dims, idata, CFL_SIZE);

		align1(tmp_dims[MAPS_DIM], tmp_dims[COIL_DIM],
				MD_CAST_ARRAY2(      complex float, DIMS, tmp_dims, tmp3, COIL_DIM, MAPS_DIM),
				MD_CAST_ARRAY2(const complex float, DIMS, tmp_dims, tmp1, COIL_DIM, MAPS_DIM),
				MD_CAST_ARRAY2(const complex float, DIMS, tmp_dims, tmp2, COIL_DIM, MAPS_DIM));

		md_copy(DIMS, tmp_dims, tmp1, tmp3, CFL_SIZE);

		md_copy_block(DIMS, (long[DIMS]){ [READ_DIM] = i + dir }, dims, odata, tmp_dims, tmp3, CFL_SIZE);
	}

	md_free(tmp1);
	md_free(tmp2);
	md_free(tmp3);
}

void align_ro(const long dims[DIMS], complex float* odata, const complex float* idata)
{
	int ro = dims[READ_DIM];
	assert(ro > 1);
#if 1
	align_ro2(dims, 0, ro, odata, idata);
#else
#pragma omp parallel sections
	{
#pragma omp section
	align_ro2(dims, ro / 2, ro, odata, idata);
#pragma omp section
	align_ro2(dims, ro / 2, -1, odata, idata);
	}
#endif
}


void gcc(const long out_dims[DIMS], complex float* out_data, const long caldims[DIMS], const complex float* cal_data)
{
	int ro = out_dims[READ_DIM];

	// zero pad calibration region along readout and FFT

	long tmp_dims[DIMS];
	md_copy_dims(DIMS, tmp_dims, caldims);
	tmp_dims[READ_DIM] = ro;
	complex float* tmp = md_alloc(DIMS, tmp_dims, CFL_SIZE);

	md_resize_center(DIMS, tmp_dims, tmp, caldims, cal_data, CFL_SIZE);
	ifftuc(DIMS, tmp_dims, READ_FLAG, tmp, tmp);

	// apply scc at each readout location

	long tmp2_dims[DIMS];
	md_select_dims(DIMS, ~READ_FLAG, tmp2_dims, tmp_dims);

	long out2_dims[DIMS];
	md_select_dims(DIMS, ~READ_FLAG, out2_dims, out_dims);


#pragma omp parallel for
	for (int i = 0; i < ro; i++) {

		complex float* tmp2 = md_alloc(DIMS, tmp2_dims, CFL_SIZE);
		complex float* out2 = md_alloc(DIMS, out2_dims, CFL_SIZE);

		long pos[DIMS] = { [READ_DIM] = i };
		md_copy_block(DIMS, pos, tmp2_dims, tmp2, tmp_dims, tmp, CFL_SIZE);

		scc(out2_dims, out2, tmp2_dims, tmp2);

		md_copy_block(DIMS, pos, out_dims, out_data, out2_dims, out2, CFL_SIZE);

		md_free(out2);
		md_free(tmp2);
	}
}



void ecc(const long out_dims[DIMS], complex float* out_data, const long caldims[DIMS], const complex float* cal_data)
{
	int channels = caldims[COIL_DIM];

	assert(1 == out_dims[PHS1_DIM]);
	assert(1 == out_dims[PHS2_DIM]);
	assert(out_dims[COIL_DIM] == channels);
	assert(out_dims[MAPS_DIM] == channels);

	struct ecalib_conf conf = ecalib_defaults;

	conf.threshold = 0.001;
	conf.crop = 0.;
	conf.kdims[0] = 6;
	conf.kdims[1] = 1;
	conf.kdims[2] = 1;
//	conf.numsv = L;
	conf.weighting = false;
	conf.orthiter = false;
	conf.perturb = 0.;

	long map_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, map_dims, out_dims);
        complex float* emaps = md_alloc(DIMS, map_dims, CFL_SIZE);

	int K = conf.kdims[0] * caldims[COIL_DIM];
	float svals[K];
	calib(&conf, out_dims, out_data, emaps, K, svals, caldims, cal_data); 

	md_free(emaps);
}



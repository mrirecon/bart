/* Copyright 2017-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2018 Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 *
 *
 * [1]
 * Kai Tobias Block and Martin Uecker, Simple Method for Adaptive
 * Gradient-Delay Compensation in Radial MRI, Annual Meeting ISMRM,
 * Montreal 2011, In Proc. Intl. Soc. Mag. Reson. Med 19: 2816 (2011)
 *
 * [2]
 * Amir Moussavi, Markus Untenberger, Martin Uecker, and Jens Frahm,
 * Correction of gradient-induced phase errors in radial MRI,
 * Magnetic Resonance in Medicine, 71:308-312 (2014)
 *
 * [3]
 * Sebastian Rosenzweig, Hans Christian Holme, Martin Uecker,
 * Simple Auto-Calibrated Gradient Delay Estimation From Few Spokes Using Radial
 * Intersections (RING), Preprint
 */

#include <complex.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#ifdef RING_PAPER
#include <stdio.h>
#endif

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/qform.h"
#include "num/fft.h"
#include "num/linalg.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/subpixel.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

// [AC-Adaptive]
static void radial_self_delays(unsigned int N, float shifts[N], const float phi[N], const long dims[DIMS], const complex float* in)
{
	unsigned int d = 2;
	unsigned int flags = (1 << d);

	assert(N == dims[d]);

	long dims1[DIMS];
	md_select_dims(DIMS, ~flags, dims1, dims);

	complex float* tmp1 = md_alloc(DIMS, dims1, CFL_SIZE);
	complex float* tmp2 = md_alloc(DIMS, dims1, CFL_SIZE);

	long pos[DIMS] = { 0 };

	for (unsigned int i = 0; i < dims[d]; i++) {

		pos[d] = i;
		md_copy_block(DIMS, pos, dims1, tmp1, dims, in, CFL_SIZE);

		// find opposing spoke

		float mdelta = 0.;
		int mindex = 0;

		for (unsigned int j = 0; j < dims[d]; j++) {

			float delta = cabsf(cexpf(1.i * phi[j]) - cexpf(1.i * phi[i]));

			if (mdelta <= delta) {

				mdelta = delta;
				mindex = j;
			}
		}

		pos[d] = mindex;
		md_copy_block(DIMS, pos, dims1, tmp2, dims, in, CFL_SIZE);


		unsigned int d2 = 1;
		float rshifts[DIMS];
		md_flip(DIMS, dims1, MD_BIT(d2), tmp2, tmp2, CFL_SIZE); // could be done by iFFT in est_subpixel_shift
		est_subpixel_shift(DIMS, rshifts, dims1, MD_BIT(d2), tmp2, tmp1);

		float mshift = rshifts[d2] / 2.; // mdelta

		shifts[i] = mshift;
	}

	md_free(tmp1);
	md_free(tmp2);
}


// [RING] Calculate Pseudoinverse
static void calc_pinv(unsigned int Nint, complex float pinv[3][2 * Nint], const complex float A[2 * Nint][3])
{
	//AH
	complex float AH[3][2 * Nint];
	mat_transpose(2 * Nint, 3, AH, A);

	// (AH * A)^-1
	complex float dot[3][3];
	mat_mul(3, 2 * Nint, 3, dot, AH, A);

	complex float inv[3][3];
	mat_inverse(3, inv, dot);

	// (AH * A)^-1 * AH
	mat_mul(3, 3, 2 * Nint, pinv, inv, AH);

	return;
}


// [RING] Find (nearly) orthogonal spokes
static void find_intersec_sp(const unsigned int no_intersec_sp, long intersec_sp[no_intersec_sp], const unsigned int cur_idx, const unsigned int N, const float angles[N])
{
	float intersec_angles[no_intersec_sp];

	for (unsigned int i = 0; i < no_intersec_sp; i++) {

		intersec_sp[i] = -1;
		intersec_angles[i] = FLT_MAX;
	}

	for (unsigned int i = 0; i < N; i++) { // Iterate through angles

		for (unsigned int j = 0; j < no_intersec_sp; j++) { // Iterate through intersec array

			// If angle difference of spoke 'i' and current spoke is greater than intersection angle 'j'

			if (fabs(fmod(fabs(angles[cur_idx] - angles[i]), M_PI) - M_PI / 2.) < intersec_angles[j]) {

				// Shift smaller intersec_angles to higher indices

				for (unsigned int k = no_intersec_sp; k > j + 1; k--) {

					intersec_sp[k - 1] = intersec_sp[k - 2]; // Spoke index
					intersec_angles[k - 1] = intersec_angles[k - 2]; // Angle value
				}

				// Assign current value
				intersec_sp[j] = i;
				intersec_angles[j] = fabs(fmod(fabs(angles[cur_idx]-angles[i]), M_PI) - M_PI / 2.); // Weight value
				break;
			}
		}
	}
}



// [RING] Test that hints if the chosen region (-r) is too small
static void check_intersections(const unsigned int Nint, const unsigned int N, const complex float S_cmplx[3][1], const float angles[N], const long idx[2][Nint], const int c_region)
{
	float phi0;
	float phi1;
	float N1;
	float N2;
	float l = 0;
	float m = 0;
	double S[3][1];

	S[0][0] = creal(S_cmplx[0][0]);
	S[1][0] = creal(S_cmplx[1][0]);
	S[2][0] = creal(S_cmplx[2][0]);

	for (unsigned int i = 0; i < Nint; i++) {

		phi0 = angles[idx[0][i]];
		phi1 = angles[idx[1][i]];
		N1 = cosf(phi0) - cosf(phi1);
		N2 = sinf(phi0) - sinf(phi1);

		// Nominal distance from spoke center to intersection point
		// (analytical formula for intersection point)

		l = (S[0][0] * N1 + S[2][0] * N2 - cosf(phi1) / sinf(phi1) * (S[2][0] * N1 + S[1][0] * N2))
			/ (cosf(phi1) * sinf(phi0) / sinf(phi1) - cosf(phi0));

		m = (S[0][0] * (-N1) + S[2][0] * (-N2) - cosf(phi0) / sinf(phi0) * (S[2][0] * (-N1) + S[1][0] * (-N2)))
			/ (cosf(phi0) * sinf(phi1) / sinf(phi0) - cosf(phi1));

		l = abs((int)round(l));
		m = abs((int)round(m));

		// Check if nominal distance can be reached in chosen region
		if ((l >= c_region / 2) || (m >= c_region / 2))
			debug_printf(DP_WARN, "Choose larger region! (-r option)\n");
	}
}



// [RING] Caclucate intersection points
static void calc_intersections(unsigned int Nint, unsigned int N, unsigned int no_intersec_sp, float dist[2][Nint], long idx[2][Nint], const float angles[N], const long kc_dims[DIMS], const complex float* kc, const int ROI)
{
	long spoke_dims[DIMS];
	md_copy_dims(DIMS, spoke_dims, kc_dims);
	spoke_dims[PHS2_DIM] = 1;

	complex float* spoke_i = md_alloc(DIMS, spoke_dims, CFL_SIZE);
	complex float* spoke_j = md_alloc(DIMS, spoke_dims, CFL_SIZE);

	long pos_i[DIMS] = { 0 };
	long pos_j[DIMS] = { 0 };

	long coilPixel_dims[DIMS];
	md_copy_dims(DIMS, coilPixel_dims, spoke_dims);
	coilPixel_dims[PHS1_DIM] = 1;

	complex float* coilPixel_l = md_alloc(DIMS, coilPixel_dims, CFL_SIZE);
	complex float* coilPixel_m = md_alloc(DIMS, coilPixel_dims, CFL_SIZE);
	complex float* diff = md_alloc(DIMS, coilPixel_dims, CFL_SIZE);

	long diff_rss_dims[DIMS];
	md_copy_dims(DIMS, diff_rss_dims, coilPixel_dims);
	diff_rss_dims[COIL_DIM] = 1;
	diff_rss_dims[PHS1_DIM] = 1;

	complex float* diff_rss = md_alloc(DIMS, diff_rss_dims, CFL_SIZE);

	long pos_l[DIMS] = { 0 };
	long pos_m[DIMS] = { 0 };

	// Array to store indices of spokes that build an angle close to pi/2 with the current spoke
	long intersec_sp[no_intersec_sp];

	for (unsigned int i = 0; i < no_intersec_sp; i++)
		intersec_sp[i] = -1;

	// Boundaries for spoke comparison
	int myROI = ROI;

	myROI += (ROI % 2 == 0) ? 1 : 0; // make odd

	int low = 0;
	int high = myROI - 1;

	int count = 0;

	// Intersection determination
	for (unsigned int i = 0; i < N; i++) {

		pos_i[PHS2_DIM] = i;

		md_slice(DIMS, PHS2_FLAG, pos_i, kc_dims, spoke_i, kc, CFL_SIZE);

		find_intersec_sp(no_intersec_sp, intersec_sp, i, N, angles);

		for (unsigned int j = 0; j < no_intersec_sp; j++) {

			pos_j[PHS2_DIM] = intersec_sp[j];

			md_slice(DIMS, PHS2_FLAG, pos_j, kc_dims, spoke_j, kc, CFL_SIZE);

			idx[0][i * no_intersec_sp + j] = i;
			idx[1][i * no_intersec_sp + j] = intersec_sp[j];

			// Elementwise rss comparisson
			float rss = FLT_MAX;

			for (int l = low; l <= high; l++) {

				pos_l[PHS1_DIM] = l;
				md_copy_block(DIMS, pos_l, coilPixel_dims, coilPixel_l, spoke_dims, spoke_i, CFL_SIZE);

				for (int m = low; m <= high; m++) {

					pos_m[PHS1_DIM] = m;

					md_copy_block(DIMS, pos_m, coilPixel_dims, coilPixel_m, spoke_dims, spoke_j, CFL_SIZE);
					md_zsub(DIMS, coilPixel_dims, diff, coilPixel_l, coilPixel_m);

					md_zrss(DIMS, coilPixel_dims, PHS1_FLAG|COIL_FLAG, diff_rss, diff);

					if (cabsf(diff_rss[0]) < rss) { // New minimum found

						rss = cabsf(diff_rss[0]);
						dist[0][i * no_intersec_sp + j] = (l + 1/2 - ROI/2);
						dist[1][i * no_intersec_sp + j] = (m + 1/2 - ROI/2);
					}
				}
			}

			count++;
		}
	}

#ifdef RING_PAPER
	// Print projection angles and corresponding offsets to files

	const char* idx_out = "projangle.txt";
	FILE* fp = fopen(idx_out, "w");

	const char* d_out = "offset.txt";
	FILE* fp1 = fopen(d_out, "w");

	for (unsigned int i = 0; i < N; i++) {

		for (unsigned int j = 0; j < no_intersec_sp; j++) {

			fprintf(fp, "%f \t %f\n", angles[idx[0][i * no_intersec_sp + j]], angles[idx[1][i * no_intersec_sp + j]]);
			fprintf(fp1, "%f \t %f\n", dist[0][i * no_intersec_sp + j], dist[1][i * no_intersec_sp + j]);
		}
	}

	fclose(fp);
	fclose(fp1);
#endif

	md_free(spoke_i);
	md_free(spoke_j);
	md_free(coilPixel_l);
	md_free(coilPixel_m);
	md_free(diff);
	md_free(diff_rss);
}



// [RING] Solve inverse problem AS = B using pseudoinverse
static void calc_S(const unsigned int Nint, const unsigned int N, complex float S[3][1], const float angles[N], const float dist[2][Nint], const long idx[2][Nint])
{
	complex float A[2 * Nint][3];
	float phi0 = 0;
	float phi1 = 0;
	float a0 = 0;
	float a1 = 0;
	float b0 = 0;
	float b1 = 0;

	complex float B[2 * Nint][1];
	complex float alpha0 = 0;
	complex float alpha1 = 0;

	unsigned int count = 0;

	for (unsigned int i = 0; i < Nint; i++) {

		phi0 = angles[idx[0][i]];
		phi1 = angles[idx[1][i]];
		a0 = cosf(phi0) - cosf(phi1);
		a1 = sinf(phi0) - sinf(phi1);

		alpha0 = dist[0][i];
		alpha1 = dist[1][i];
		b0 = alpha1 * cosf(phi1) - alpha0 * cosf(phi0);
		b1 = alpha1 * sinf(phi1) - alpha0 * sinf(phi0);

		A[count][0] = a0;
		A[count][1] = 0.;
		A[count][2] = a1;
		B[count][0] = b0;
		count++;

		A[count][0] = 0.;
		A[count][1] = a1;
		A[count][2] = a0;
		B[count][0] = b1;
		count++;
	}

	complex float pinv[3][2 * Nint];

	calc_pinv(Nint, pinv, A);
	mat_mul(3, 2 * Nint, 1, S, pinv, B);
}


static const char usage_str[] = "<trajectory> <data>";
static const char help_str[] = "Estimate gradient delays from radial data.";


int main_estdelay(int argc, char* argv[])
{
	bool ring = false;
	int pad_factor = 100;
	unsigned int no_intersec_sp = 1;
	float size = 1.5;

	const struct opt_s opts[] = {

		OPT_SET('R', &ring, "RING method"),
		OPT_INT('p', &pad_factor, "p", "[RING] Padding"),
		OPT_UINT('n', &no_intersec_sp, "n", "[RING] Number of intersecting spokes"),
		OPT_FLOAT('r', &size, "r", "[RING] Central region size"),
	};

	cmdline(&argc, argv, 2, 2, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (pad_factor % 2 != 0)
		error("Pad_factor -p should be even\n");


	long tdims[DIMS];
	const complex float* traj = load_cfl(argv[1], DIMS, tdims);

	long tdims1[DIMS];
	md_select_dims(DIMS, ~MD_BIT(1), tdims1, tdims);

	complex float* traj1 = md_alloc(DIMS, tdims1, CFL_SIZE);
	md_slice(DIMS, MD_BIT(1), (long[DIMS]){ 0 }, tdims, traj1, traj, CFL_SIZE);

	unsigned int N = tdims[2];

	float angles[N];
	for (unsigned int i = 0; i < N; i++)
		angles[i] = M_PI + atan2f(crealf(traj1[3 * i + 0]), crealf(traj1[3 * i + 1]));


	if (ring) {

		assert(0 == tdims[1] % 2);

		md_slice(DIMS, MD_BIT(1), (long[DIMS]){ [1] = tdims[1] / 2 }, tdims, traj1, traj, CFL_SIZE);

		for (unsigned int i = 0; i < N; i++)
			if (0. != cabsf(traj1[3 * i]))
				error("Nominal trajectory must be centered for RING.\n");
	}


	md_free(traj1);


	long full_dims[DIMS];
	const complex float* full_in = load_cfl(argv[2], DIMS, full_dims);

	// Remove not needed dimensions
	long dims[DIMS];
	md_select_dims(DIMS, READ_FLAG|PHS1_FLAG|PHS2_FLAG|COIL_FLAG, dims, full_dims);
	complex float* in = md_alloc(DIMS, dims, CFL_SIZE);

	long pos[DIMS] = { 0 };
	md_copy_block(DIMS, pos, dims, in, full_dims, full_in, CFL_SIZE);

	// FIXME: more checks
	assert(dims[1] == tdims[1]);
	assert(dims[2] == tdims[2]);

	float delays[N];

	if (!ring) { // AC-Adaptive method [1]

		radial_self_delays(N, delays, angles, dims, in);

		/* We allow an arbitrary quadratic form to account for
		 * non-physical coordinate systems.
		 * Moussavi et al., MRM 71:308-312 (2014)
		 */

		float qf[3];
		fit_quadratic_form(qf, N, angles, delays);
		bart_printf("%f:%f:%f\n", qf[0], qf[1], qf[2]);

	} else { // RING method [3]

		int c_region = (int)(pad_factor * size);

		// Make odd to have 'Center of c_region' == 'DC component of spoke'
		c_region += (c_region % 2 == 0) ? 1 : 0;

		//--- Refine grid ---
		complex float* im = md_alloc(DIMS, dims, CFL_SIZE);
		ifftuc(DIMS, dims, PHS1_FLAG, im, in);

		long pad_dims[DIMS];
		md_copy_dims(DIMS, pad_dims, dims);
		pad_dims[PHS1_DIM] = pad_factor * dims[PHS1_DIM];

		complex float* k_pad = md_alloc(DIMS, pad_dims, CFL_SIZE);
		complex float* im_pad = md_alloc(DIMS, pad_dims, CFL_SIZE);

		md_resize_center(DIMS, pad_dims, im_pad, dims, im, CFL_SIZE);
		md_free(im);

		// Sinc filter in k-space (= crop FOV in image space)
		long crop_dims[DIMS];
		md_copy_dims(DIMS, crop_dims, dims);
		crop_dims[PHS1_DIM] = 0.6 * dims[PHS1_DIM];

		complex float* crop = md_alloc(DIMS, crop_dims, CFL_SIZE);
		md_zfill(DIMS, crop_dims, crop, 1.);

		complex float* mask = md_alloc(DIMS, pad_dims, CFL_SIZE);
		md_resize_center(DIMS, pad_dims, mask, crop_dims, crop, CFL_SIZE);

		complex float* im_pad2 = md_alloc(DIMS, pad_dims, CFL_SIZE);
		md_zmul(DIMS, pad_dims, im_pad2, im_pad, mask);
		md_free(im_pad);
		md_free(mask);

		fftuc(DIMS, pad_dims, PHS1_FLAG, k_pad, im_pad2);
		md_free(im_pad2);

		//--- Consider only center region ---

		long kc_dims[DIMS];
		md_copy_dims(DIMS, kc_dims, pad_dims);
		kc_dims[PHS1_DIM] = c_region;

		complex float* kc = md_alloc(DIMS, kc_dims, CFL_SIZE);

		long pos[DIMS] = { 0 };
		pos[PHS1_DIM] = pad_dims[PHS1_DIM] / 2 - (c_region / 2);
		md_copy_block(DIMS, pos, kc_dims, kc, pad_dims, k_pad, CFL_SIZE);
		md_free(k_pad);

		//--- Calculate intersections ---

		unsigned int Nint = N * no_intersec_sp; // Number of intersection points
		long idx[2][Nint];
		float dist[2][Nint];
		complex float S[3][1] = { { 0 } };

		calc_intersections(Nint, N, no_intersec_sp, dist, idx, angles, kc_dims, kc, c_region);
		calc_S(Nint, N, S, angles, dist, idx);
		check_intersections(Nint, N, S, angles, idx, c_region);

		bart_printf("%f:%f:%f\n\n", creal(S[0][0]) / pad_factor, creal(S[1][0]) / pad_factor, creal(S[2][0]) / pad_factor);
		md_free(kc);
	}

	unmap_cfl(DIMS, full_dims, full_in);
	unmap_cfl(DIMS, tdims, traj);
	md_free(in);

	return 0;
}



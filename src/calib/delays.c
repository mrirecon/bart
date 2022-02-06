/* Copyright 2017-2019. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * Sebastian Rosenzweig <sebastian.rosenzweig@med.uni-goettingen.de>
 *
 *
 * Kai Tobias Block and Martin Uecker, Simple Method for Adaptive
 * Gradient-Delay Compensation in Radial MRI, Annual Meeting ISMRM,
 * Montreal 2011, In Proc. Intl. Soc. Mag. Reson. Med 19: 2816 (2011)
 *
 * Amir Moussavi, Markus Untenberger, Martin Uecker, and Jens Frahm,
 * Correction of gradient-induced phase errors in radial MRI,
 * Magnetic Resonance in Medicine, 71:308-312 (2014)
 *
 * Sebastian Rosenzweig, Hans Christian Holme, Martin Uecker,
 * Simple Auto-Calibrated Gradient Delay Estimation From Few Spokes Using Radial
 * Intersections (RING), Magnetic Resonance in Medicine 81:1898-1906 (2019)
 */

#include <complex.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/linalg.h"

#include "misc/debug.h"
#include "misc/subpixel.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/version.h"

#include "delays.h"


// [AC-Adaptive]
void radial_self_delays(int N, float shifts[N], const float phi[N], const long dims[DIMS], const complex float* in)
{
	int d = 2;
	int flags = (1 << d);

	assert(N == dims[d]);

	long dims1[DIMS];
	md_select_dims(DIMS, ~flags, dims1, dims);

	complex float* tmp1 = md_alloc(DIMS, dims1, CFL_SIZE);
	complex float* tmp2 = md_alloc(DIMS, dims1, CFL_SIZE);

	long pos[DIMS] = { 0 };

	for (int i = 0; i < dims[d]; i++) {

		pos[d] = i;
		md_copy_block(DIMS, pos, dims1, tmp1, dims, in, CFL_SIZE);

		// find opposing spoke

		float mdelta = 0.;
		int mindex = 0;

		for (int j = 0; j < dims[d]; j++) {

			float delta = cabsf(cexpf(1.i * phi[j]) - cexpf(1.i * phi[i]));

			if (mdelta <= delta) {

				mdelta = delta;
				mindex = j;
			}
		}

		pos[d] = mindex;
		md_copy_block(DIMS, pos, dims1, tmp2, dims, in, CFL_SIZE);


		int d2 = 1;
		float rshifts[DIMS];
		md_flip(DIMS, dims1, MD_BIT(d2), tmp2, tmp2, CFL_SIZE); // could be done by iFFT in est_subpixel_shift
		est_subpixel_shift(DIMS, rshifts, dims1, MD_BIT(d2), tmp2, tmp1);

		float mshift = rshifts[d2] / 2.; // mdelta

		shifts[i] = mshift;
	}

	md_free(tmp1);
	md_free(tmp2);
}


static float angle_dist(float a, float b)
{
	return fabsf(fmodf(fabsf(a - b), M_PI) - (float)(M_PI / 2.));
}



static void find_nearest_orthogonal_spokes(int N, int spokes[N], float ref_angle, const float angles[N])
{
	float dist[N];
	__block float* distp = dist; // clang workaround

	for (int i = 0; i < N; i++) {

		spokes[i] = i;
		dist[i] = angle_dist(ref_angle, angles[i]);
	}

	NESTED(int, dist_compare, (int a, int b))
	{
		float d = distp[a] - distp[b];

		if (d > 0.)
			return 1;

		return (0. == d) ? 0 : -1;
	};

	quicksort(N, spokes, dist_compare);
}



// [RING] Find (nearly) orthogonal spokes
static void find_intersec_sp(const int no_intersec_sp, int intersec_sp[no_intersec_sp], const int cur_idx, const int N, const float angles[N])
{
	float intersec_angles[no_intersec_sp];

	for (int i = 0; i < no_intersec_sp; i++) {

		intersec_sp[i] = -1;
		intersec_angles[i] = FLT_MAX;
	}

	for (int i = 0; i < N; i++) { // Iterate through angles

		for (int j = 0; j < no_intersec_sp; j++) { // Iterate through intersec array

			// If angle difference of spoke 'i' and current spoke is greater than intersection angle 'j'

			if (fabs(fmod(fabs(angles[cur_idx] - angles[i]), M_PI) - M_PI / 2.) < intersec_angles[j]) {

				// Shift smaller intersec_angles to higher indices

				for (int k = no_intersec_sp; k > j + 1; k--) {

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
static void check_intersections(const int Nint, const int N, const float S[3], const float angles[N], const long idx[Nint][2], const int c_region)
{
	for (int i = 0; i < Nint; i++) {

		float phi0 = angles[idx[i][0]];
		float phi1 = angles[idx[i][1]];

		float N1 = cosf(phi0) - cosf(phi1);
		float N2 = sinf(phi0) - sinf(phi1);

		// Nominal distance from spoke center to intersection point
		// (analytical formula for intersection point)

		float l = (S[0] * N1 + S[2] * N2 - cosf(phi1) / sinf(phi1) * (S[2] * N1 + S[1] * N2))
				/ (cosf(phi1) * sinf(phi0) / sinf(phi1) - cosf(phi0));

		float m = (S[0] * (-N1) + S[2] * (-N2) - cosf(phi0) / sinf(phi0) * (S[2] * (-N1) + S[1] * (-N2)))
				/ (cosf(phi0) * sinf(phi1) / sinf(phi0) - cosf(phi1));

		l = abs((int)round(l));
		m = abs((int)round(m));

		// Check if nominal distance can be reached in chosen region
		if ((l >= c_region / 2) || (m >= c_region / 2)) {

			debug_printf(DP_WARN, "Choose larger region! (-r option)\n");
			break;
		}
	}
}



// [RING] Caclucate intersection points
static void calc_intersections(int Nint, int N, int no_intersec_sp, float dist[Nint][2], long idx[Nint][2], const float angles[N], const long kc_dims[DIMS], const complex float* kc)
{
	long spoke_dims[DIMS];
	md_select_dims(DIMS, ~PHS2_FLAG, spoke_dims, kc_dims);

	complex float* spoke_i = md_alloc(DIMS, spoke_dims, CFL_SIZE);
	complex float* spoke_j = md_alloc(DIMS, spoke_dims, CFL_SIZE);

	long pos_i[DIMS] = { 0 };
	long pos_j[DIMS] = { 0 };

	int ROI = kc_dims[PHS1_DIM];

	// Intersection determination
	for (int i = 0; i < N; i++) {

		pos_i[PHS2_DIM] = i;

		md_slice(DIMS, PHS2_FLAG, pos_i, kc_dims, spoke_i, kc, CFL_SIZE);

		int intersec_sp[N];
		find_nearest_orthogonal_spokes(N, intersec_sp, angles[i], angles);

		if (use_compat_to_version("v0.4.00")) // for RING reproducibility
			find_intersec_sp(no_intersec_sp, intersec_sp, i, N, angles);

		for (int j = 0; j < no_intersec_sp; j++) {

			pos_j[PHS2_DIM] = intersec_sp[j];

			md_slice(DIMS, PHS2_FLAG, pos_j, kc_dims, spoke_j, kc, CFL_SIZE);

			idx[i * no_intersec_sp + j][0] = i;
			idx[i * no_intersec_sp + j][1] = intersec_sp[j];

			// Elementwise rss comparisson
			float ss = FLT_MAX;
			int channels = spoke_dims[COIL_DIM];

			for (int l = 0; l < ROI; l++) {

				for (int m = 0; m < ROI; m++) {

					float diff_ss = 0.;

					for (int c = 0; c < channels; c++) {

						complex float diff = spoke_i[l + c * ROI] - spoke_j[m + c * ROI];

						diff_ss += pow(crealf(diff), 2.) + pow(cimagf(diff), 2.);
					}

					if (diff_ss < ss) { // New minimum found

						ss = diff_ss;
						dist[i * no_intersec_sp + j][0] = (l + 1/2 - ROI/2);
						dist[i * no_intersec_sp + j][1] = (m + 1/2 - ROI/2);
					}
				}
			}
		}
	}


	// Print projection angles and corresponding offsets, for RING paper reproduction
	{
		char* str = getenv("RING_PAPER");

		if ( (NULL != str) && (1 == atoi(str))) {

			for (int i = 0; i < N; i++) {

				for (int j = 0; j < no_intersec_sp; j++) {

					bart_printf("projangle: %f \t %f\n", angles[idx[i * no_intersec_sp + j][0]], angles[idx[i * no_intersec_sp + j][1]]);
					bart_printf("offset: %f \t %f\n", dist[i * no_intersec_sp + j][0], dist[i * no_intersec_sp + j][1]);
				}
			}
		}
	}

	md_free(spoke_i);
	md_free(spoke_j);
}



// [RING] Solve inverse problem AS = B using pseudoinverse
static void calc_S(const int Nint, const int N, float S[3], const float angles[N], const float dist[Nint][2], const long idx[Nint][2])
{
	complex float A[2 * Nint][3];
	complex float B[2 * Nint];

	for (int i = 0; i < Nint; i++) {

		float phi0 = angles[idx[i][0]];
		float phi1 = angles[idx[i][1]];

		float a0 = cosf(phi0) - cosf(phi1);
		float a1 = sinf(phi0) - sinf(phi1);

		float b0 = dist[i][1] * cosf(phi1) - dist[i][0] * cosf(phi0);
		float b1 = dist[i][1] * sinf(phi1) - dist[i][0] * sinf(phi0);

		A[2 * i + 0][0] = a0;
		A[2 * i + 0][1] = 0.;
		A[2 * i + 0][2] = a1;
		B[2 * i + 0] = b0;

		A[2 * i + 1][0] = 0.;
		A[2 * i + 1][1] = a1;
		A[2 * i + 1][2] = a0;
		B[2 * i + 1] = b1;
	}

	complex float pinv[3][2 * Nint];

	mat_pinv_left(2 * Nint, 3, pinv, A);

	complex float Sc[3];

	mat_vecmul(3, 2 * Nint, Sc, pinv, B);

	for (int i = 0; i < 3; i++)
		S[i] = crealf(Sc[i]);
}


struct ring_conf ring_defaults = {

	.pad_factor = 100,
	.size = 1.5,
	.no_intersec_sp = 1,
	.crop_factor = 0.6,
};

void ring(const struct ring_conf* conf, float S[3], int N, const float angles[N], const long dims[DIMS], const complex float* in)
{
	assert(dims[2] == N);

	int c_region = (int)(conf->pad_factor * conf->size);

	// Make odd to have 'Center of c_region' == 'DC component of spoke'
	if (0 == c_region % 2)
		c_region++;

	complex float* im = md_alloc(DIMS, dims, CFL_SIZE);

	ifftuc(DIMS, dims, PHS1_FLAG, im, in);


	// Sinc filter in k-space (= crop FOV in image space)

	long crop_dims[DIMS];

	md_copy_dims(DIMS, crop_dims, dims);
	crop_dims[PHS1_DIM] = conf->crop_factor * dims[PHS1_DIM];

	complex float* crop = md_alloc(DIMS, crop_dims, CFL_SIZE);

	md_resize_center(DIMS, crop_dims, crop, dims, im, CFL_SIZE);

	md_free(im);

	//--- Refine grid ---

	long pad_dims[DIMS];

	md_copy_dims(DIMS, pad_dims, dims);
	pad_dims[PHS1_DIM] = conf->pad_factor * dims[PHS1_DIM];

	complex float* pad = md_alloc(DIMS, pad_dims, CFL_SIZE);

	md_resize_center(DIMS, pad_dims, pad, crop_dims, crop, CFL_SIZE);

	md_free(crop);

	fftuc(DIMS, pad_dims, PHS1_FLAG, pad, pad);


	//--- Consider only center region ---

	long kc_dims[DIMS];

	md_copy_dims(DIMS, kc_dims, pad_dims);
	kc_dims[PHS1_DIM] = c_region;

	complex float* kc = md_alloc(DIMS, kc_dims, CFL_SIZE);

	long pos[DIMS] = { 0 };
	pos[PHS1_DIM] = pad_dims[PHS1_DIM] / 2 - (c_region / 2);

	md_copy_block(DIMS, pos, kc_dims, kc, pad_dims, pad, CFL_SIZE);

	md_free(pad);

	//--- Calculate intersections ---

	int Nint = N * conf->no_intersec_sp; // Number of intersection points
	long idx[Nint][2];
	float dist[Nint][2];

	calc_intersections(Nint, N, conf->no_intersec_sp, dist, idx, angles, kc_dims, kc);

	calc_S(Nint, N, S, angles, dist, idx);

	check_intersections(Nint, N, S, angles, idx, c_region);

	for (int i = 0; i < 3; i++)
		S[i] /= conf->pad_factor;

	md_free(kc);
}

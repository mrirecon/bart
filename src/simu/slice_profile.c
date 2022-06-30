/* Copyright 2022. TU Graz. Institute for Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Author:
 *	Nick Scholand
 */

#include <complex.h>
#include <math.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/fft.h"

#include "misc/mri.h"
#include "misc/debug.h"

#include "simu/pulse.h"
#include "simu/bloch.h"
#include "simu/simulation.h"

#include "slice_profile.h"



/* Approximate slice-profile with Fourier transform of the RF-pulses envelope
 *
 * Pauly J, Nishimura D, Macovski A.
 * A k-space analysis of small-tip-angle excitation.
 * J Magn Reson. 1989;81:43-56.
 *
 * ! This function only takes the main lope into account !
 * ! Be careful to have a high enough BWTP !
 */

// FIXME: How to create a utest for it?!
void slice_profile_fourier(int N, const long dims[N], complex float* out, const struct simdata_pulse* pulse)
{
	float samples = 1000.;
	float dt = (pulse->rf_end - pulse->rf_start) / samples;

	long pulse_dims[DIMS];
	md_set_dims(DIMS, pulse_dims, 1);

	pulse_dims[READ_DIM] = samples;

	complex float* envelope = md_alloc(DIMS, pulse_dims, CFL_SIZE);

	for (int i = 0; i < samples; i++)
		envelope[i] = pulse_sinc(pulse, pulse->rf_start + i * dt);


	// Zero-pad for increased frequency sampling rate

	long pad_dims[DIMS];
	md_copy_dims(DIMS, pad_dims, pulse_dims);

	pad_dims[READ_DIM] = 6 * pulse_dims[READ_DIM];	// 6 seems to allow accurate frequency sampling

	complex float* padded = md_alloc(DIMS, pad_dims, CFL_SIZE);

	md_resize_center(DIMS, pad_dims, padded, pulse_dims, envelope, CFL_SIZE);


	// Determine Slice Profile

	complex float* slice_profile = md_alloc(DIMS, pad_dims, CFL_SIZE);

	fftc(DIMS, pad_dims, READ_FLAG, slice_profile, padded);


	// Find maximum in Slice Profile amplitude and scale it to 1

	float amp_max = 0.;

	for (int i = 0; i < pad_dims[READ_DIM]; i++)
		amp_max = (cabsf(slice_profile[i]) > amp_max) ? cabsf(slice_profile[i]) : amp_max;

	assert(0. < amp_max);
	debug_printf(DP_DEBUG3, "Max Amplitude of Slice Profile %f\n", amp_max);

	md_zsmul(DIMS, pad_dims, slice_profile, slice_profile, 1. / amp_max);


	// Threshold to find slice frequency limits

	float limit = 0.01;	//Limit from which slice is taken into account

	int count = 0;

	for (long i = 0; i < pad_dims[READ_DIM]; i++) {

		if (cabsf(slice_profile[i]) > limit)
			count++;
		else
			slice_profile[i] = 0.;
	}

	assert(0 < count);


	// Separate counted elements

	long count_dims[DIMS];
	md_set_dims(DIMS, count_dims, 1);

	count_dims[READ_DIM] = count;

	complex float* slice_count = md_alloc(DIMS, count_dims, CFL_SIZE);

	for (long i = 0; i < count_dims[READ_DIM]; i++)
		slice_count[i] = slice_profile[(pad_dims[READ_DIM] - count) / 2 + i + count % 2]; //count%2 compensates for integer division error for odd `count`


	// Linear interpolation of final slice profile samples

	int slcprfl_samples = dims[READ_DIM] * 2;

	long slc_sample_dims[DIMS];
	md_set_dims(DIMS, slc_sample_dims, 1);

	slc_sample_dims[READ_DIM] = slcprfl_samples;

	complex float* slc_samples = md_alloc(DIMS, slc_sample_dims, CFL_SIZE);

	float steps = (float)(count_dims[READ_DIM] + 1) / (float)slcprfl_samples;	// +1 because of zero indexing of count_dims

	for (int i = 0; i < slcprfl_samples; i++) {

		int ppos = (int) (i * steps);

		float pdiv =  (i * steps) - ppos;

		assert(0 <= pdiv);

		int npos = ppos + 1;

		slc_samples[i] = (slice_count[npos] - slice_count[ppos]) * pdiv + slice_count[ppos];

		debug_printf(DP_DEBUG1, "SLICE SAMPLES: i: %d,\t (i * steps): %f,\t ppos: %d,\t pdiv: %f,\t npos: %d,\tslice sample: %f\n", i, (i * steps), ppos, pdiv, npos, cabsf(slc_samples[i]));
	}

	// Copy desired amount of Slice Profile Samples

	for (int i = 0; i < dims[READ_DIM]; i++)
		out[i] = slc_samples[i];

	md_free(envelope);
	md_free(padded);
	md_free(slice_profile);
	md_free(slice_count);
	md_free(slc_samples);
}


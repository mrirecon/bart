/* Copyright 2020. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2018-2020 Sebastian Rosenzweig
 * 2020 Martin Uecker
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/filter.h"
#include "num/vecops.h"

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "bin.h"

const struct bin_conf_s bin_defaults = {

	.n_resp = 0,
	.n_card = 0,
	.mavg_window = 0,
	.mavg_window_card = 0,
	.cluster_dim = -1,

	.resp_labels_idx = { 0, 1 },
	.card_labels_idx = { 2, 3 },

	.card_out = NULL,

	.offset_angle = { 0., 0. },

	.amplitude = 0,

};

// Binning by equal central angle
static void det_bins(const complex float* state, const long bins_dims[DIMS], float* bins, const int idx, const int n, float offset)
{
	int T = bins_dims[TIME_DIM];

	float central_angle = 2. * M_PI / n;

	for (int t = 0; t < T; t++) {

		float angle = atan2f(crealf(state[T + t]), crealf(state[t])) + offset * ( 2 * M_PI / 360.);

		angle = (angle < 0.) ? (angle + 2. * M_PI) : angle;

		bins[idx * T + t] = floorf(angle / central_angle);

 		//debug_printf(DP_INFO, "%f: bin %f\n", (M_PI + atan2f(crealf(state[T + t]), crealf(state[t]))) * 360 / 2. / M_PI, bins[idx * T + t]);
	}
}

// Binning by amplitude
static void det_bins_amp(const long state_dims[DIMS], const complex float* state, const long bins_dims[DIMS], float* bins, const int idx, const int n)
{
	int T = bins_dims[TIME_DIM];
	
	float* s = md_alloc(DIMS, state_dims, FL_SIZE);
	
	md_real(DIMS, state_dims, s, state);

	float min = quickselect(s, T, T - 1); // resorts s!

	md_real(DIMS, state_dims, s, state);
	md_sadd(DIMS, state_dims, s, s, -min); // make positive

	float max = quickselect(s, T, 0); // resorts s!

	md_real(DIMS, state_dims, s, state);
	md_sadd(DIMS, state_dims, s, s, -min);

	float delta = (float)max / n;
	float amp = 0.;

	for (int t = 0; t < T; t++) {

		amp = s[t];
		bins[idx * T + t] = floorf(amp * 0.99 / delta);
	}

	md_free(s);
}



/* Check if time is consistent with increasing bin index
 *
 * Idea: Calculate the angles defined by EOF_a & EOF_b (phase diagram!) for time
 * steps total_time / 2 and total_time / 2 + 1. If the angle increases, time evolution
 * is consistent with increasing bin-index.  Otherwise, swap EOF_a with EOF_b.
 */
static bool check_valid_time(const long singleton_dims[DIMS], complex float* singleton, const long labels_dims[DIMS], const complex float* labels, const long labels_idx[2])
{
	// Indices at half of total time
	int idx_0 = floor(singleton_dims[TIME_DIM] / 2.);
	int idx_1 = idx_0 + 1;

	long pos[DIMS] = { 0 };

	pos[TIME2_DIM] = labels_idx[0];
	md_copy_block(DIMS, pos, singleton_dims, singleton, labels_dims, labels, CFL_SIZE);

	float a_0 = crealf(singleton[idx_0]);
	float a_1 = crealf(singleton[idx_1]);

	pos[TIME2_DIM] = labels_idx[1];
	md_copy_block(DIMS, pos, singleton_dims, singleton, labels_dims, labels, CFL_SIZE);

	float b_0 = crealf(singleton[idx_0]);
	float b_1 = crealf(singleton[idx_1]);

	float angle_0 = atan2f(b_0, a_0);

	if (angle_0 < 0.)
		angle_0 += 2. * M_PI;

	float angle_1 = atan2f(b_1, a_1);

	if (angle_1 < 0.)
		angle_1 += 2. * M_PI;

	// Check if angle increases (and consider phase wrap!)
	float diff = angle_1 - angle_0;

	return ((diff >= 0.) == (fabsf(diff) <= M_PI));
}


// Calculate maximum number of samples in a bin
static int get_binsize_max(const long bins_dims[DIMS], const float* bins, const int n_card, const int n_resp)
{
	// Array to count number of appearances of a bin
	long count_dims[2] = { n_card, n_resp };

	int* count = md_calloc(2, count_dims, sizeof(int));

	int T = bins_dims[TIME_DIM]; // Number of time samples

	for (int t = 0; t < T; t++) { // Iterate through time

		int cBin = (int)bins[0 * T + t];
		int rBin = (int)bins[1 * T + t];

		count[rBin * n_card + cBin]++;
	}

	// Determine value of array maximum
	int binsize_max = 0;

	for (int r = 0; r < n_resp; r++) {

		for (int c = 0; c < n_card; c++) {

			if (count[r * n_card + c] > binsize_max)
				binsize_max = count[r * n_card + c];

			//debug_printf(DP_INFO, "%d\n", count[r * n_card + c]);
		}
	}

	md_free(count);

	return binsize_max;
}




static void moving_average(const long state_dims[DIMS], complex float* state, const int mavg_window)
{
	// Pad with boundary values
	long pad_dims[DIMS];
	md_copy_dims(DIMS, pad_dims, state_dims);

	pad_dims[TIME_DIM] = state_dims[TIME_DIM] + mavg_window -1;

	complex float* pad = md_alloc(DIMS, pad_dims, CFL_SIZE);

	md_resize_center(DIMS, pad_dims, pad, state_dims, state, CFL_SIZE);

	long singleton_dims[DIMS];
	md_select_dims(DIMS, TIME2_FLAG, singleton_dims, state_dims);

	complex float* singleton = md_alloc(DIMS, singleton_dims, CFL_SIZE);

	long pos[DIMS] = { 0 };
	md_copy_block(DIMS, pos, singleton_dims, singleton, state_dims, state, CFL_SIZE); // Get first value of array

	long start = labs((pad_dims[TIME_DIM] / 2) - (state_dims[TIME_DIM] / 2));

	for (int i = 0; i < start; i++) { // Fill beginning of pad array

		pos[TIME_DIM] = i;
		md_copy_block(DIMS, pos, pad_dims, pad, singleton_dims, singleton, CFL_SIZE);
	}

	long end = mavg_window - start;

	pos[TIME_DIM] = state_dims[TIME_DIM] - 1;
	md_copy_block(DIMS, pos, singleton_dims, singleton, state_dims, state, CFL_SIZE); // Get last value of array

	for (int i = 0; i < end; i++) { // Fill end of pad array

		pos[TIME_DIM] = pad_dims[TIME_DIM] - 1 - i;
		md_copy_block(DIMS, pos, pad_dims, pad, singleton_dims, singleton, CFL_SIZE);
	}

	// Calc moving average
	long tmp_dims[DIMS + 1];
	md_copy_dims(DIMS, tmp_dims, pad_dims);
	tmp_dims[DIMS] = 1;

	long tmp_strs[DIMS + 1];
	md_calc_strides(DIMS, tmp_strs, tmp_dims, CFL_SIZE);

	tmp_dims[TIME_DIM] = state_dims[TIME_DIM]; // Moving-average-reduced temporal dimension

	long tmp2_strs[DIMS + 1];
	md_calc_strides(DIMS + 1, tmp2_strs, tmp_dims, CFL_SIZE);

	tmp_dims[DIMS] = mavg_window;
	tmp_strs[DIMS] = tmp_strs[TIME_DIM];

	complex float* mavg = md_alloc(DIMS, tmp_dims, CFL_SIZE);

	md_zavg2(DIMS + 1, tmp_dims, (1u << DIMS), tmp2_strs, mavg, tmp_strs, pad);
	md_zsub(DIMS, state_dims, state, state, mavg);

	md_free(pad);
	md_free(mavg);
	md_free(singleton);
}




extern int bin_quadrature(const long bins_dims[DIMS], float* bins,
			const long labels_dims[DIMS], complex float* labels,
			const struct bin_conf_s conf)
{
	// Extract respiratory labels
	long resp_state_dims[DIMS];
	md_copy_dims(DIMS, resp_state_dims, labels_dims);
	resp_state_dims[TIME2_DIM] = 2;

	complex float* resp_state = md_alloc(DIMS, resp_state_dims, CFL_SIZE);

	long resp_state_singleton_dims[DIMS];
	md_copy_dims(DIMS, resp_state_singleton_dims, resp_state_dims);
	resp_state_singleton_dims[TIME2_DIM] = 1;

	complex float* resp_state_singleton = md_alloc(DIMS, resp_state_singleton_dims, CFL_SIZE);

	bool valid_time_resp = check_valid_time(resp_state_singleton_dims, resp_state_singleton, labels_dims, labels, conf.resp_labels_idx);

	long pos[DIMS] = { 0 };

	for (int i = 0; i < 2; i++){

		pos[TIME2_DIM] = conf.resp_labels_idx[i];
		md_copy_block(DIMS, pos, resp_state_singleton_dims, resp_state_singleton, labels_dims, labels, CFL_SIZE);

		if (valid_time_resp)
			pos[TIME2_DIM] = i;
		else
			pos[TIME2_DIM] = 1 - i;

		md_copy_block(DIMS, pos, resp_state_dims, resp_state, resp_state_singleton_dims, resp_state_singleton, CFL_SIZE);
	}


	// Extract cardiac labels
	long card_state_dims[DIMS];
	md_copy_dims(DIMS, card_state_dims, labels_dims);
	card_state_dims[TIME2_DIM] = 2;

	complex float* card_state = md_alloc(DIMS, card_state_dims, CFL_SIZE);

	long card_state_singleton_dims[DIMS];
	md_copy_dims(DIMS, card_state_singleton_dims, card_state_dims);
	card_state_singleton_dims[TIME2_DIM] = 1;

	complex float* card_state_singleton = md_alloc(DIMS, card_state_singleton_dims, CFL_SIZE);

	bool valid_time_card = check_valid_time(card_state_singleton_dims, card_state_singleton, labels_dims, labels, conf.card_labels_idx);

	for (int i = 0; i < 2; i++) {

		pos[TIME2_DIM] = conf.card_labels_idx[i];
		md_copy_block(DIMS, pos, card_state_singleton_dims, card_state_singleton, labels_dims, labels, CFL_SIZE);

		if (valid_time_card)
			pos[TIME2_DIM] = i;
		else // If time evolution is not consistent with increasing bin-index, swap order of the two EOFs
			pos[TIME2_DIM] = 1 - i;

		md_copy_block(DIMS, pos, card_state_dims, card_state, card_state_singleton_dims, card_state_singleton, CFL_SIZE);
	}

	if (conf.mavg_window > 0) {

		moving_average(resp_state_dims, resp_state, conf.mavg_window);
		moving_average(card_state_dims, card_state, (conf.mavg_window_card > 0) ? conf.mavg_window_card : conf.mavg_window);
	}

	if (NULL != conf.card_out)
		dump_cfl(conf.card_out, DIMS, card_state_dims, card_state);

	// Determine bins
	if (conf.amplitude)
		det_bins_amp(resp_state_dims, resp_state, bins_dims, bins, 1, conf.n_resp); // amplitude binning for respiratory motion
	else
		det_bins(resp_state, bins_dims, bins, 1, conf.n_resp, conf.offset_angle[0]); // respiratory motion	 

	det_bins(card_state, bins_dims, bins, 0, conf.n_card, conf.offset_angle[1]); // cardiac motion

	md_free(card_state);
	md_free(card_state_singleton);

	md_free(resp_state);
	md_free(resp_state_singleton);

	return get_binsize_max(bins_dims, bins, conf.n_card, conf.n_resp);
}



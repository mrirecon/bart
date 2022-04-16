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
#include <stdio.h>

#include "num/multind.h"
#include "num/init.h"
#include "num/flpmath.h"
#include "num/filter.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "calib/bin.h"


/* Reorder binning: [-o]
 * --------------------
 *
 * Input a 1D file with <labels> at the <src> dimension that
 * you want to reorder according to the label order.
 *
 *
 * Label binning: [-l long]
 * ------------------------
 *
 * Bin a dimension according to the label-file <bin-signal>
 * The label file must be 1D and the dimension of the <label> file
 * and the <src> file must match at the dimension that you want to bin
 * -d must specify an empty dimension, in which the binned data is stored.
 *
 *
 * Quadrature binning:
 * -------------------
 *
 * Sebastian Rosenzweig, Nick Scholand, H. Christian M. Holme, Martin Uecker.
 * Cardiac and Respiratory Self-Gating in Radial MRI using an Adapted Singular Spectrum
 * Analysis (SSA-FARY). IEEE Tran Med Imag 2020; 10.1109/TMI.2020.2985994. arXiv:1812.09057.
 *
 * n: Number of bins
 * state: Contains amplitudes for both EOFs
 * idx: Bins for 'idx' motion (0: cardiac, 1: respiration)
 */








// Copy spokes from input array to correct position in output array
static void asgn_bins(const long bins_dims[DIMS], const float* bins, const long sg_dims[DIMS], complex float* sg, const long in_dims[DIMS], const complex float* in, const int
n_card, const int n_resp)
{
	// Array to keep track of numbers of spokes already asigned to each bin
	long count_dims[2] = { n_card, n_resp };

	int* count = md_calloc(2, count_dims, sizeof(int));


	// Array to store a single spoke (including read-out, [coils] and slices)
	long in_singleton_dims[DIMS];
	md_select_dims(DIMS, ~TIME_FLAG, in_singleton_dims, in_dims);

	complex float* in_singleton = md_alloc(DIMS, in_singleton_dims, CFL_SIZE);

	int T = bins_dims[TIME_DIM]; // Number of time samples

	long pos0[DIMS] = { 0 };
	long pos1[DIMS] = { 0 };

	for (int t = 0; t < T; t++) { // Iterate all spokes of input array

		pos0[TIME_DIM] = t;
		md_copy_block(DIMS, pos0, in_singleton_dims, in_singleton, in_dims, in, CFL_SIZE);

		int cBin = (int)bins[0 * T + t];
		int rBin = (int)bins[1 * T + t];

		pos1[PHS2_DIM] = count[rBin * n_card + cBin]; // free spoke index in respective bin
		pos1[TIME_DIM] = cBin;
		pos1[TIME2_DIM] = rBin;

		md_copy_block(DIMS, pos1, sg_dims, sg, in_singleton_dims, in_singleton, CFL_SIZE);

		count[rBin * n_card + cBin]++;
	}

	md_free(in_singleton);
	md_free(count);
}



static int find_dim(int N, long dims[N])
{
	int dim = -1;
	int count = 0;

	for (int i = 0; i < N; i++) {

		if (dims[i] > 1) {

			dim = i;
			count++;
		}
	}

	assert(count == 1);
	return dim;
}





static const char help_str[] = "Binning";



int main_bin(int argc, char* argv[argc])
{
	bool reorder = false;
	bool amplitude = false;
	struct bin_conf_s conf = bin_defaults;
	
	const char* label_file = NULL;
	const char* src_file = NULL;
	const char* dst_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &label_file, "label"),
		ARG_INFILE(true, &src_file, "src"),
		ARG_OUTFILE(true, &dst_file, "dst"),
	};

	const struct opt_s opts[] = {

		OPT_INT('l', &conf.cluster_dim, "dim", "Bin according to labels: Specify cluster dimension"),
		OPT_SET('o', &reorder, "Reorder according to labels"),
		OPT_UINT('R', &conf.n_resp, "n_resp", "Quadrature Binning: Number of respiratory labels"),
		OPT_UINT('C', &conf.n_card, "n_card", "Quadrature Binning: Number of cardiac labels"),
		OPT_VEC2('r', &conf.resp_labels_idx, "x:y", "(Respiration: Eigenvector index)"),
		OPT_VEC2('c', &conf.card_labels_idx, "x:y", "(Cardiac motion: Eigenvector index)"),
		OPT_UINT('a', &conf.mavg_window, "window", "Quadrature Binning: Moving average"),
		OPT_UINT('A', &conf.mavg_window_card, "window", "(Quadrature Binning: Cardiac moving average window)"),
		OPT_FLVEC2('O', &conf.offset_angle, "[r:c]deg", "Quadrature Binning: Angle offset for resp and card."),
		OPT_STRING('x', &conf.card_out, "file", "(Output filtered cardiac EOFs)"), // To reproduce SSA-FARY paper
		OPT_SET('M', &conf.amplitude, "Amplitude binning"),

	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	// Input
	long labels_dims[DIMS];
	complex float* labels = load_cfl(label_file, DIMS, labels_dims);

	long src_dims[DIMS];
	complex float* src = load_cfl(src_file, DIMS, src_dims);

	enum { BIN_QUADRATURE, BIN_LABEL, BIN_REORDER } bin_type;

	if (amplitude) {
		
		assert(conf.cluster_dim != -1);
		assert(conf.mavg_window == 0);
	}

	// Identify binning type
	if ((conf.n_resp > 0) || (conf.n_card > 0)) {

		debug_printf(DP_INFO, "Quadrature binning...\n");

		bin_type = BIN_QUADRATURE;

		assert((conf.n_resp > 0) && (conf.n_card > 0));
		assert(conf.cluster_dim == -1);
		assert(!reorder);

	} else if (conf.cluster_dim != -1) {

		debug_printf(DP_INFO, "Label binning...\n");

		bin_type = BIN_LABEL;;

		if ((conf.cluster_dim < 0) || (src_dims[conf.cluster_dim] != 1)) // Dimension to store data for each cluster must be empty
			error("Choose empty cluster dimension!");

		assert((conf.n_resp == 0) && (conf.n_card == 0));
		assert(!reorder);

	} else if (reorder) {

		debug_printf(DP_INFO, "Reorder binning...\n");

		bin_type = BIN_REORDER;

		assert((conf.n_resp == 0) && (conf.n_card == 0));
		assert(conf.cluster_dim == -1);
		assert(reorder);

	} else {

		error("No bin type specified!");
	}



	switch (bin_type) {

	case BIN_QUADRATURE:

		if (labels_dims[TIME_DIM] < 2)
			error("Check dimensions of labels array!");

		// Array to store bin-index for samples
		long bins_dims[DIMS];
		md_copy_dims(DIMS, bins_dims, labels_dims);
		bins_dims[TIME2_DIM] = 2; // Respiration and Cardiac motion

		float* bins = md_alloc(DIMS, bins_dims, FL_SIZE);

		int binsize_max = bin_quadrature(bins_dims, bins, labels_dims, labels, conf);

		long binned_dims[DIMS];
		md_copy_dims(DIMS, binned_dims, src_dims);
		binned_dims[TIME_DIM] = conf.n_card;
		binned_dims[TIME2_DIM] = conf.n_resp;
		binned_dims[PHS2_DIM] = binsize_max;

		complex float* binned = create_cfl(dst_file, DIMS, binned_dims);
		md_clear(DIMS, binned_dims, binned, CFL_SIZE);

		asgn_bins(bins_dims, bins, binned_dims, binned, src_dims, src, conf.n_card, conf.n_resp);

		md_free(bins);

		unmap_cfl(DIMS, binned_dims, binned);

		break;

	case BIN_REORDER: // Reorder: Assign to dst from src according to labels
	case BIN_LABEL: // Label binning: Bin elements from src according to labels

		md_check_compat(DIMS, ~0u, src_dims, labels_dims);
		md_check_bounds(DIMS, ~0u, labels_dims, src_dims);

		int dim = find_dim(DIMS, labels_dims); // Dimension to be binned
		int N = labels_dims[dim]; // number of samples to be binned

		// Determine number of clusters
		int n_clusters = 0;

		for (int i = 0; i < N; i++) {

			int label = (int)crealf(labels[i]);

			assert(0 <= label);

			n_clusters = MAX(n_clusters, label);
		}

		n_clusters++; // Account for zero-based indexing


		// Determine all cluster sizes
		int cluster_size[n_clusters];

		for (int i = 0; i < n_clusters; i++)
			cluster_size[i] = 0;

		for (int i = 0; i < N; i++)
			cluster_size[(int)crealf(labels[i])]++;

		// Determine maximum cluster size
		int cluster_max = 0;

		for (int i = 0; i < n_clusters; i++)
			cluster_max = MAX(cluster_max, cluster_size[i]);

		// Initialize output
		long dst_dims[DIMS];
		md_copy_dims(DIMS, dst_dims, src_dims);

		if (BIN_REORDER != bin_type) {

			dst_dims[conf.cluster_dim] = cluster_max;
			dst_dims[dim] = n_clusters;

		} else {

			dst_dims[dim] = labels_dims[dim];
			assert(-1 == conf.cluster_dim);
			assert(n_clusters <= src_dims[dim]);
		}


		complex float* dst = create_cfl(dst_file, DIMS, dst_dims);

		md_clear(DIMS, dst_dims, dst, CFL_SIZE);


		long singleton_dims[DIMS];
		md_select_dims(DIMS, ~MD_BIT(dim), singleton_dims, src_dims);

		complex float* singleton = md_alloc(DIMS, singleton_dims, CFL_SIZE);

		int idx[n_clusters];

		for (int i = 0; i < n_clusters; i++)
			idx[i] = 0;

		long pos_src[DIMS] = { 0 };
		long pos_dst[DIMS] = { 0 };

		for (int i = 0; i < N; i++) { // TODO: Speed but by direct copying

			int label = (int)crealf(labels[i]);

			if (BIN_REORDER != bin_type) {

				pos_src[dim] = i;
				pos_dst[dim] = label;

				pos_dst[conf.cluster_dim] = idx[label]; // Next empty singleton index for i-th cluster
				idx[label]++;

			} else {

				pos_src[dim] = label;	// switched!
				pos_dst[dim] = i;
			}

			md_copy_block(DIMS, pos_src, singleton_dims, singleton, src_dims, src, CFL_SIZE);
			md_copy_block(DIMS, pos_dst, dst_dims, dst, singleton_dims, singleton, CFL_SIZE);

			if (0 == i % (N / 10))
				debug_printf(DP_DEBUG3, "Binning: %f%\n", 100. * i / N);
		}

		md_free(singleton);

		unmap_cfl(DIMS, dst_dims, dst);

		break;

	} // end switch case

	unmap_cfl(DIMS, labels_dims, labels);
	unmap_cfl(DIMS, src_dims, src);

	return 0;
}


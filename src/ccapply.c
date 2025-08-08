/* Copyright 2016-2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2017 Jonathan Tamir
 */

#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/stream.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "calib/cc.h"




static const char help_str[] = "Apply coil compression forward/inverse operation.";



int main_ccapply(int argc, char* argv[argc])
{
	const char* ksp_file = NULL;
	const char* cc_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &ksp_file, "kspace"),
		ARG_INFILE(true, &cc_file, "cc_matrix"),
		ARG_OUTFILE(true, &out_file, "proj_kspace"),
	};

	bool forward = true;
	bool do_fft = true;
	long P = -1;
	enum cc_type { SCC, GCC, ECC } cc_type = SCC;
	int aligned = -1;

	const struct opt_s opts[] = {

		OPT_LONG('p', &P, "N", "perform compression to N virtual channels"),
		OPT_CLEAR('u', &forward, "apply inverse operation"),
		OPT_CLEAR('t', &do_fft, "don't apply FFT in readout"),
		OPT_SELECT('S', enum cc_type, &cc_type, SCC, "type: SVD"),
		OPT_SELECT('G', enum cc_type, &cc_type, GCC, "type: Geometric"),
		OPT_SELECT('E', enum cc_type, &cc_type, ECC, "type: ESPIRiT"),
		OPT_PINT('A', &aligned, "dim", "Perform alignment of coil sensitivities along dimension A"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long in_dims[DIMS];
	long cc_dims[DIMS];

	complex float* in_data = NULL;
	complex float* cc_data = NULL;

	long in_dims_t[DIMS];
	md_set_dims(DIMS, in_dims_t, 0);

	long cc_dims_t[DIMS];

	complex float* in_data_t = NULL;
	complex float* cc_data_t = NULL;

	stream_t strm_in = NULL;
	stream_t strm_cc = NULL;

	const unsigned long rtflags = 1024;


	if (-1 != aligned) {

		assert(10 == aligned);
		assert(cc_type == SCC);

		in_data_t = load_async_cfl(ksp_file, DIMS, in_dims_t);
		cc_data_t = load_async_cfl(cc_file, DIMS, cc_dims_t);

		assert(in_dims_t[TIME_DIM] == cc_dims_t[TIME_DIM]);
		md_select_dims(DIMS, ~TIME_FLAG, in_dims, in_dims_t);
		md_select_dims(DIMS, ~TIME_FLAG, cc_dims, cc_dims_t);

		strm_in = stream_lookup(in_data_t);
		strm_cc = stream_lookup(cc_data_t);

	} else {

		in_data = load_cfl(ksp_file, DIMS, in_dims);
		cc_data = load_cfl(cc_file, DIMS, cc_dims);
	}


	assert(1 == in_dims[MAPS_DIM]);
	const long channels = cc_dims[COIL_DIM];

	if (-1 == P)
		P = in_dims[COIL_DIM];

	assert(cc_dims[MAPS_DIM] >= P && in_dims[COIL_DIM] >= P);

	long out_dims[DIMS] = { [0 ... DIMS - 1] = 1 };

	md_select_dims(DIMS, ~COIL_FLAG, out_dims, in_dims);
	out_dims[COIL_DIM] = forward ? P : channels;

	complex float* out_data = NULL;


	long out_dims_t[DIMS];
	md_copy_dims(DIMS, out_dims_t, out_dims);

	if (-1 != aligned)
		out_dims_t[TIME_DIM] = in_dims_t[TIME_DIM];

	complex float* out_data_t = NULL;

	stream_t strm_out = NULL;

	if (-1 != aligned) {

		assert((NULL == strm_in) || (stream_get_flags(strm_in) == rtflags));
		assert((NULL == strm_out) || (stream_get_flags(strm_out) == rtflags));

		out_data_t = create_async_cfl(out_file, rtflags, DIMS, out_dims_t);
		strm_out = stream_lookup(out_data_t);

	} else {

		out_data = create_cfl(out_file, DIMS, out_dims);
	}


	// transpose for the matrix multiplication
	long trp_dims[DIMS];

	if (forward) {

		debug_printf(DP_DEBUG1, "Compressing to %ld virtual coils...\n", P);

		md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, trp_dims, out_dims);
		trp_dims[MAPS_DIM] = out_dims[COIL_DIM];

	} else {

		debug_printf(DP_DEBUG1, "Uncompressing channels...\n");

		md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, trp_dims, in_dims);
	}

	long cc2_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cc2_dims, cc_dims);
	cc2_dims[MAPS_DIM] = P;


	long pos[DIMS];
	long cc2_dims_t[DIMS];
	long in_str_t[DIMS];
	long cc_str_t[DIMS];
	long cc2_str_t[DIMS];
	long out_str_t[DIMS];

	complex float* rt_cc2_data = NULL;
	complex float* rt_tmp = NULL;

	if (-1 != aligned) {

		md_set_dims(DIMS, pos, 0);
		md_copy_dims(DIMS, cc2_dims_t, cc2_dims);
		cc2_dims_t[TIME_DIM] = in_dims_t[TIME_DIM];

		md_calc_strides(DIMS, in_str_t, in_dims_t, CFL_SIZE);
		md_calc_strides(DIMS, cc_str_t, cc_dims_t, CFL_SIZE);
		md_calc_strides(DIMS, out_str_t, out_dims_t, CFL_SIZE);
		md_calc_strides(DIMS, cc2_str_t, cc2_dims, CFL_SIZE);

		rt_cc2_data = anon_cfl(NULL, DIMS, cc2_dims_t);
		rt_tmp = md_alloc(DIMS, cc2_dims, CFL_SIZE);

		if (strm_cc)
			stream_sync(strm_cc, DIMS, pos);

		md_copy_block(DIMS, (long [DIMS]){ }, cc2_dims, rt_tmp, cc_dims, cc_data_t, CFL_SIZE);
	}

	if (SCC != cc_type) {

		if (do_fft) {

			complex float* in2_data = anon_cfl(NULL, DIMS, in_dims);

			ifftuc(DIMS, in_dims, READ_FLAG, in2_data, in_data);

			unmap_cfl(DIMS, in_dims, in_data);

			in_data = in2_data;
		}

		complex float* cc2_data = anon_cfl(NULL, DIMS, cc2_dims);

		align_ro(cc2_dims, cc2_data, cc_data);

		unmap_cfl(DIMS, cc_dims, cc_data);

		cc_data = cc2_data;
	}

rt_loop:
	if (-1 != aligned) {

		if (strm_cc)
			stream_sync(strm_cc, DIMS, pos);
		if (strm_in)
			stream_sync(strm_in, DIMS, pos);

		complex float* cc_data_unaligned = &MD_ACCESS(DIMS, cc_str_t, pos, cc_data_t);

		in_data =  &MD_ACCESS(DIMS, in_str_t, pos, in_data_t);
		out_data = &MD_ACCESS(DIMS, out_str_t, pos, out_data_t);
		cc_data =  &MD_ACCESS(DIMS, cc2_str_t, pos, rt_cc2_data);

		cc_align_mat(cc2_dims, cc_data, cc_data_unaligned, rt_tmp);

		md_copy_block(DIMS, (long [DIMS]){ }, cc2_dims, rt_tmp, cc_dims, cc_data, CFL_SIZE);
	}


	if (forward)
		md_zmatmulc(DIMS, trp_dims, out_data, cc2_dims, cc_data, in_dims, in_data);
	else
		md_zmatmul(DIMS, out_dims, out_data, cc2_dims, cc_data, trp_dims, in_data);


	if (-1 != aligned) {

		if (strm_out)
			stream_sync(strm_out, DIMS, pos);

		if (md_next(DIMS, in_dims_t, TIME_FLAG, pos))
			goto rt_loop;

		unmap_cfl(DIMS, cc2_dims_t, rt_cc2_data);
		md_free(rt_tmp);
	}

	if (SCC != cc_type) {

		if (do_fft)
			fftuc(DIMS, out_dims, READ_FLAG, out_data, out_data);

		if (-1 != aligned)
			unmap_cfl(DIMS, cc_dims_t, cc_data_t);
		else
			unmap_cfl(DIMS, cc2_dims, cc_data);

	} else {

		if (-1 != aligned)
			unmap_cfl(DIMS, cc_dims_t, cc_data_t);
		else
			unmap_cfl(DIMS, cc_dims, cc_data);
	}

	if (-1 != aligned) {

		unmap_cfl(DIMS, in_dims_t, in_data_t);
		unmap_cfl(DIMS, out_dims_t, out_data_t);

	} else {

		unmap_cfl(DIMS, in_dims, in_data);
		unmap_cfl(DIMS, out_dims, out_data);
	}

	debug_printf(DP_DEBUG1, "Done.\n");

	return 0;
}


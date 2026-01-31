/* Copyright 2016-2018. Martin Uecker.
 * Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/resize.h"
#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "misc/stream.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


static const char help_str[] = "Copy an array (to a given position in the output file - which then must exist).";

static void delay_seconds(double seconds)
{
	int nanoseconds = (seconds - trunc(seconds)) * 1.E9;

	struct timespec sleep_spec = { .tv_sec = seconds, .tv_nsec = nanoseconds };
	nanosleep(&sleep_spec, NULL);
}


int main_copy(int argc, char* argv[argc])
{
	int count = 0;
	long* dims = NULL;
	long* poss = NULL;
	unsigned long stream_flags = 0UL;

	const char* in_file = NULL;
	const char* out_file = NULL;

	float delay = 0.;

	struct arg_s args[] = {

		ARG_TUPLE(false, &count, 2, TUPLE_LONG(&dims, "dim"),
					    TUPLE_LONG(&poss, "pos")),
		ARG_INFILE(true, &in_file, "input"),
		ARG_INOUTFILE(true, &out_file, "output"),
	};

	struct opt_s opts[] = {

		OPTL_ULONG(0, "stream", &stream_flags, "flags", "Loop over <flags> while streaming."),
		OPTL_FLOAT(0, "delay", &delay, "f", "Wait for f seconds before each copy when streaming."),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();


	bool is_stream = 0 == stream_flags ? false : true;

	int N = DIMS;

	assert(count >= 0);
	assert((0 == count) || (!is_stream));

	long in_dims[N];
	long out_dims[N];

	complex float* in_data = (is_stream ? load_async_cfl : load_cfl)(in_file, N, in_dims);

	if (count > 0) {

		if (FILE_TYPE_CFL != file_type(out_file))
			error("Output file must be a cfl if position is specified.\n");

		// get dimensions
		complex float* out_data = load_cfl(out_file, N, out_dims);

		unmap_cfl(N, out_dims, out_data);

		io_close(out_file);

	} else {

		md_copy_dims(N, out_dims, in_dims);
	}

	complex float* out_data = NULL;

	if (!is_stream)
		out_data = create_cfl(out_file, N, out_dims);
	else
		out_data = create_async_cfl(out_file, stream_flags, N, out_dims);

	long position[N];

	for (int i = 0; i < N; i++)
		position[i] = 0;

	for (int i = 0; i < count; i++) {

		long dim = dims[i];
		long pos = poss[i];

		assert(dim < N);
		assert((0 <= pos) && (pos < out_dims[dim]));

		position[dim] = pos;
	}

	long stream_pos[N];
	long non_stream_idims[N];
	long non_stream_odims[N];
	md_set_dims(N, stream_pos, 0);

	// these can both be non-Null despite 0 == stream_flags, because of looping.
	stream_t strm_in = stream_lookup(in_data);
	stream_t strm_out = stream_lookup(out_data);

	md_select_dims(N, ~stream_flags, non_stream_odims, out_dims);
	md_select_dims(N, ~stream_flags, non_stream_idims, in_dims);

	long ostr[N];
	md_calc_strides(N, ostr, out_dims, CFL_SIZE);

	long istr[N];
	md_calc_strides(N, istr, in_dims, CFL_SIZE);

	do {
		if (is_stream && strm_in)
			stream_sync_slice(strm_in, N, in_dims, stream_flags, stream_pos);

		md_copy_block2(N, position, non_stream_odims, ostr, &MD_ACCESS(N, ostr, stream_pos, out_data),
					    non_stream_idims, istr, &MD_ACCESS(N, istr, stream_pos, in_data), CFL_SIZE);

		delay_seconds(delay);

		if (is_stream && strm_out)
			stream_sync_slice(strm_out, N, out_dims, stream_flags, stream_pos);

	} while (md_next(N, in_dims, stream_flags, stream_pos));

	unmap_cfl(N, in_dims, in_data);
	unmap_cfl(N, out_dims, out_data);

	xfree(dims);
	xfree(poss);

	return 0;
}


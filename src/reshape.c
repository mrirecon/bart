/* Copyright 2015-2016. The Regents of the University of California.
 * Copyright 2015-2021. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2021-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/stream.h"
#include "misc/debug.h"

#define DIMS 16

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

static const char help_str[] = "Reshape selected dimensions.";


int main_reshape(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	int count = 0;
	long* dims = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "flags"),
		ARG_TUPLE(true, &count, 1, TUPLE_LONG(&dims, "dim")),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	unsigned long stream_flags = 0 ;

	const struct opt_s opts[] = { 

		OPT_ULONG('s', &stream_flags, "flags", "stream flagged dims"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	int n = bitcount(flags);

	assert(n == count);

	long in_dims[DIMS];
	long out_dims[DIMS];

	complex float* in_data = (0 != stream_flags ? load_async_cfl : load_cfl)(in_file, DIMS, in_dims);

	md_copy_dims(DIMS, out_dims, in_dims);
	
	int j = 0;

	long otot = 1;
	long itot = 1;

	for (int i = 0; i < DIMS; i++) {

		if (MD_IS_SET(flags, i)) {

			otot *= dims[j];
			itot *= in_dims[i];
			out_dims[i] = dims[j++];
		}
	}

	if (otot != itot) {

		debug_print_dims(DP_INFO, DIMS, in_dims);
		debug_print_dims(DP_INFO, DIMS, out_dims);
		error("Reshaped dimensions are incompatible!\n");
	}

	assert(j == n);

	complex float* out_data = NULL;
	
	if (0 == stream_flags) {

		out_data = create_cfl(out_file, DIMS, out_dims);

		md_reshape(DIMS, flags, out_dims, out_data, in_dims, in_data, CFL_SIZE);

	} else {

		out_data = create_async_cfl(out_file, stream_flags, DIMS, out_dims);

		stream_t strm_in = stream_lookup(in_data);
		stream_t strm_out = stream_lookup(out_data);

		unsigned long iflags = 0;
		unsigned long oflags = 0;

		if (NULL != strm_in)
			iflags = stream_get_flags(strm_in);

		oflags = stream_get_flags(strm_out);

		if (0 != (~flags & (iflags | oflags)))
			error("All streamd dimensions must be reshaped!");

		long slc_dims[DIMS];
		md_select_dims(DIMS, ~flags, slc_dims, in_dims);

		void* buf = md_alloc(DIMS, slc_dims, CFL_SIZE);

		long ipos[DIMS] = { };
		long opos[DIMS] = { };

		bool stream_loop = (strm_in && !cfl_loop_desc_active());
		long idx_count = 0;

		do {
			if (stream_loop && !stream_receive_pos(strm_in, idx_count++, DIMS, ipos))
				break;

			do {
				long index = md_ravel_index(DIMS, ipos, flags, in_dims);
				md_unravel_index(DIMS, opos, flags, out_dims, index);

				md_slice(DIMS, flags, ipos, in_dims, buf, in_data, CFL_SIZE);

				long zpos[DIMS] = { };
				md_move_block(DIMS, slc_dims, opos, out_dims, out_data, zpos, slc_dims, buf, CFL_SIZE);

				bool cont = false;

				for (int i = 0; i < DIMS; i++) {

					if (!MD_IS_SET(flags, i))
						continue;

					if (!MD_IS_SET(oflags, i) && (1 + opos[i] != out_dims[i]))
						cont = true;
				}

				if (cont)
					continue;

				long opos2[DIMS];
				md_select_strides(DIMS, oflags, opos2, opos);
				stream_sync_slice(strm_out, DIMS, out_dims, oflags, opos2);

			} while (md_next(DIMS, in_dims, flags & ~iflags, ipos));

			for (int i = 0; i < DIMS; i++)
				if (MD_IS_SET(flags & ~iflags, i))
					ipos[i] = 0;

		} while (stream_loop || md_next(DIMS, in_dims, iflags, ipos));

		md_free(buf);
	}

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);
	xfree(dims);

	return 0;
}


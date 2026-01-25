/* Copyright 2024-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Philip Schaten
 */

#include <complex.h>
#include <stdbool.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/init.h"

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


static const char help_str[] = "Transmit/Receive a stream in binary format.";


int main_trx(int argc, char* argv[argc])
{
	const char *infile = "-";
	const char *outfile = "-";

	struct arg_s args[] = {};
	struct opt_s opts[] = {

		OPT_INFILE('i', &infile, "<infile>", "Infile (stdin)"),
		OPT_OUTFILE('o', &outfile, "<output>", "Outfile (stdout)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (cfl_loop_desc_active())
		error("trx tool is not compatible with looping.\n");

	int N = DIMS;
	long dims[N];

	complex float* data = NULL;
	unsigned long stream_flags = 0;
	bool bin_in = false;

	stream_t strm_in = NULL;
	stream_t strm_out = NULL;

	// first, check if input data is binary stream
	if (FILE_TYPE_PIPE == file_type(infile)) {

		char *dataname;
		strm_in = stream_load_file(infile, N, dims, &dataname);

		if (stream_is_binary(strm_in)) {

			assert(NULL == dataname);
			bin_in = true;

		} else {

			data = shared_cfl(N, dims, dataname);

			if (0 != unlink(dataname))
				error("Error unlinking temporary file %s\n", dataname);

			free(dataname);

			stream_attach(strm_in, data, true, true);
		}

		stream_flags = stream_get_flags(strm_in);

	} else {

		data = load_cfl(infile, N, dims);
	}

	// create output accordingly
	if (bin_in) {

		data = create_async_cfl(outfile, stream_flags, N, dims);
		strm_out = stream_lookup(data);

		stream_attach(strm_in, data, false, false);

	} else {

		strm_out = stream_create_file(outfile, N, dims, stream_flags, NULL, false);

		stream_attach(strm_out, data, false, false);
	}

	assert(strm_out);

	long stream_pos[N];
	md_set_dims(N, stream_pos, 0);
	long count = 0;

	do {
		if (strm_in && !stream_receive_serial(strm_in, N, stream_pos, count++))
			break;

		stream_sync(strm_out, N, stream_pos);

	} while (strm_in || md_next(N, dims, stream_flags, stream_pos));

	unmap_cfl(N, dims, data);

	if (bin_in)
		stream_free(strm_in);
	else
		stream_free(strm_out);

	return 0;
}


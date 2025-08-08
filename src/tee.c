/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2023 Moritz Blumenthal <blumenthal@tugraz.at>
 * 2023 Philip Schaten <philip.schaten@tugraz.at>
 */

#include <complex.h>
#include <signal.h>
#include <stdbool.h>

#include "misc/debug.h"
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


static const char help_str[] = "Copy stdin to stdout + given output files.";


static void stream_sigpipe_handler(int /*signum*/)
{
}

int main_tee(int argc, char* argv[argc])
{
	int count = 0;
	const char** out_files = NULL;
	const char *out0 = NULL;
	bool no_stdout = false;
	bool keep_going = true;
	const char *in_file = "-";

	struct arg_s args[] = {

		ARG_TUPLE(false, &count, 1, { OPT_STRING, sizeof(char*), &out_files, "write to files (and stdout)" }),
	};

	bool timer = false;

	struct opt_s opts[] = {

		OPTL_STRING('i', "in", &in_file, "data", "Input File (instead of stdin)"),
		OPTL_OUTFILE('\0', "out0", &out0, "meta", "Output file which receives only metadata"),
		OPT_SET('t', &timer, "print time between inputs"),
		OPT_SET('n', &no_stdout, "No stdout"),
		OPT_CLEAR('a', &keep_going, "Abort program on disappearing inputs/outputs."),
	};

	double time = timestamp();

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	if (cfl_loop_desc_active())
		error("Currently, bart loop is not compatible with tee.\n");

	mmio_file_locking = false;

	if (keep_going) {

		// sigpipe can occur when a receiving program closes a pipe early.
		// tee explicitly allows that and therefore needs to ignore sigpipe.
		struct sigaction old_sigaction = {};
		sigaction(SIGPIPE, &(struct sigaction){ .sa_handler = stream_sigpipe_handler }, &old_sigaction);

		// Make sure we don't overwrite any other handler.
		// this does not work if the program is run from a systemd unit.
		// assert((SIG_DFL == old_sigaction.sa_handler) || (stream_sigpipe_handler == old_sigaction.sa_handler));
	}

	long dims[DIMS];


	if (0 == strcmp("-", in_file)) {

		if (no_stdout)
			io_reserve_input(in_file);
		else
			io_reserve_inout(in_file);
	} else {

		io_reserve_input(in_file);
		if (!no_stdout)
			io_reserve_output("-");
	}


	complex float* in_data = load_async_cfl(in_file, DIMS, dims);

	stream_t stream_in = stream_lookup(in_data);
	unsigned long stream_flags = stream_in ? stream_get_flags(stream_in) : 0;

	if (NULL != out0) {

		complex float* out0_addr = create_async_cfl(out0, stream_flags, DIMS, dims);
		stream_t strm = stream_lookup(out0_addr);

		if (strm) {

			stream_sync_slice_try(strm, DIMS, dims, 0, (long [DIMS]){ 0 });
			stream_free(strm);
		}

		unmap_cfl(DIMS, dims, out0_addr);
	}

	int files_offset = no_stdout ? 0 : 1;

	count += files_offset;

	long slice_dims[DIMS];
	md_select_dims(DIMS, ~stream_flags, slice_dims, dims);

#pragma omp parallel for num_threads(count)
	for (int i = 0; i < count; i++) {

		const char* name = ((0 == i) && !no_stdout) ? "-" : out_files[i - files_offset];

		if (0 != strcmp(name, "-"))
			io_reserve_output(name);

		complex float* out_data = create_async_cfl(name, stream_flags, DIMS, dims);

		long pos[DIMS];
		md_set_dims(DIMS, pos, 0);
		long counter = 0;

		do {

			if (stream_in && !stream_receive_pos(stream_in, counter, DIMS, pos))
				break;

			md_move_block(DIMS, slice_dims, pos, dims, out_data, pos, dims, in_data, CFL_SIZE);

			stream_t stream_out = stream_lookup(out_data);

			// Output streams may disappear if the receiving process needs
			// only part of the data.
			if (stream_out && !stream_sync_slice_try(stream_out, DIMS, dims, stream_flags, pos)) {

				stream_free(stream_out);
				// if for whatever reason we didn't get sigpipe, terminate anyways if not keep_going
				if (!keep_going)
					error("tee: output vanished\n");

				break;
			}

			if (timer)
				fprintf(stderr, "frame %ld: %fs\n", counter, timestamp() - time);

			time = timestamp();

			counter++;

		} while (stream_in || md_next(DIMS, dims, stream_flags, pos));

		unmap_cfl(DIMS, dims, out_data);
	}

	unmap_cfl(DIMS, dims, in_data);

	if (out_files)
		xfree(out_files);

	return 0;
}


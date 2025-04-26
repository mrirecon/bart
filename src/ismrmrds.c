/* Copyright 2021-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h" 
#include "misc/debug.h" 
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"
#include "misc/stream.h"


#include "ismrm/read.h"
#include "ismrm/xml_wrapper.h"

static const char help_str[] = "Read ISMRMRD Stream.";


int main_ismrmrds(int argc, char* argv[argc])
{
	const char* in_file = "-";
	const char* out_file = NULL;

	struct isrmrm_config_s config = ismrm_default_config;

	struct arg_s args[] = {

		ARG_OUTFILE(false, &out_file, "output"),
	};

	const struct opt_s opts[] = {};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	config.ismrm_cpp_state = ismrm_stream_open(in_file);

	long D = DIMS;
	long dims[D];
	md_set_dims(D, dims, 0);

	ismrm_stream_read_dims(&config, D, dims);

	unsigned long flags = 0;
	for (int i = 1; i < D; i++)
		if (1 < dims[i])
			flags |= MD_BIT(i);

	long pos[D];
	md_set_dims(D, pos, 0);

	complex float* out = create_async_cfl(out_file ? : "-", flags, D, dims);
	stream_t s_out = stream_lookup(out);

	while (ismrm_stream_read(&config, D, dims, pos, out))
		if(s_out)
			stream_sync_slice(s_out, D, dims, flags, pos);

	unmap_cfl(D, dims, out);

	ismrm_stream_close(config.ismrm_cpp_state);

	return 0;
}


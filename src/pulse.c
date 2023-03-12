/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/mmio.h"

#include "simu/pulse.h"


static const char help_str[] = "Pulse generation tool";
int main_pulse(int argc, char* argv[argc])
{
	const char* out_signal = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_signal, "Signal: Bxy"),
	};

	const struct opt_s opts[] = {

	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	int N = 256;

	struct pulse_sinc ps = pulse_sinc_defaults;

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);
	dims[0] = N;

	complex float* pulse = create_cfl(out_signal, DIMS, dims);

	for (int i = 0; i < N; i++)
		pulse[i] = pulse_sinc(&ps, (i + 0.5) * ps.INTERFACE.duration / N);

	unmap_cfl(DIMS, dims, pulse);

	return 0;
}



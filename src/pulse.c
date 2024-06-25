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

	enum pulse_type {
		PULSE_SINC, PULSE_HYPSEC, PULSE_RECT
	} pulse_type = PULSE_SINC;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_signal, "Signal: Bxy"),
	};

	const struct opt_s opts[] = {

		OPTL_SELECT(0, "sinc", enum pulse_type, &pulse_type, PULSE_SINC, "sinc"),
		OPTL_SELECT(0, "rect", enum pulse_type, &pulse_type, PULSE_RECT, "rect"),
		OPTL_SELECT(0, "hypsec", enum pulse_type, &pulse_type, PULSE_HYPSEC, "hypersecant"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	int N = 256;
	const struct pulse* ps = NULL;

	switch (pulse_type) {
	case PULSE_SINC: ps = CAST_UP(&pulse_sinc_defaults); break;
	case PULSE_HYPSEC: ps = CAST_UP(&pulse_hypsec_defaults); break;
	case PULSE_RECT: ps = CAST_UP(&pulse_rect_defaults); break;
	}

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);
	dims[0] = N;

	complex float* pulse = create_cfl(out_signal, DIMS, dims);

	for (int i = 0; i < N; i++)
		pulse[i] = pulse_eval(ps, (i + 0.5) * ps->duration / N);

	unmap_cfl(DIMS, dims, pulse);

	return 0;
}



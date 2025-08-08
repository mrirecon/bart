/* Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <assert.h>
#include <math.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/debug.h"
#include "misc/opts.h"
#include "misc/mri.h"
#include "misc/mmio.h"

#include "seq/pulse.h"


static const char help_str[] = "Pulse generation tool";

int main_pulse(int argc, char* argv[argc])
{
	const char* out_signal = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_signal, "Signal: Bxy"),
	};

	enum pulse_t pulse_type = PULSE_SINC;

	int Ntime = -1;

	double dur = 0.001;
	double flip_angle = 90.0;
	double bwtp = 4.;
	int mb = 1;
	double sms_dist = 0.01;
	double slice_th = 0.005;


	const struct opt_s opts[] = {

		OPTL_SELECT(0, "sinc", enum pulse_t, &pulse_type, PULSE_SINC, "sinc"),
		OPTL_SELECT(0, "sms", enum pulse_t, &pulse_type, PULSE_SINC_SMS, "sms"),
		OPTL_SELECT(0, "rect", enum pulse_t, &pulse_type, PULSE_REC, "rect"),
		OPTL_SELECT(0, "hypsec", enum pulse_t, &pulse_type, PULSE_HS, "hypersecant"),

		/* Pulse Specific Parameters */
		OPTL_DOUBLE(0, "dur", &dur, "double", "pulse duration [s]"),
		OPTL_DOUBLE(0, "fa", &flip_angle, "double", "flip angle [deg]"),
		OPTL_DOUBLE(0, "bwtp", &bwtp, "double", "bandwidth time product"),
		OPTL_PINT(0, "mb", &mb, "long", "SMS multi-band factor"),
		OPTL_DOUBLE(0, "sms-dist", &sms_dist, "long", "center-to-center slice distance between SMS partitions [m]"),
		OPTL_DOUBLE(0, "slice-th", &slice_th, "double", "slice thickness [m]"),
		OPTL_PINT(0, "N", &Ntime, "int", "number of time-steps (default = 1e6 * dur)"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if ((1 != mb) && (PULSE_SINC_SMS != pulse_type)) {

		pulse_type = PULSE_SINC_SMS;
		debug_printf(DP_INFO, "pulse type set to PULSE_SINC_SMS\n");
	}

	if (-1 == Ntime)
		Ntime = ceilf(1.e6 * dur);

	struct pulse_sinc ps = pulse_sinc_defaults;
	struct pulse_sms pm[mb];
	struct pulse_hypsec ph = pulse_hypsec_defaults;
	struct pulse_rect pr = pulse_rect_defaults;

	struct pulse* pulse[mb];

	switch (pulse_type) {

	case PULSE_SINC:

		pulse_sinc_init(&ps, dur, flip_angle, 0, bwtp, pulse_sinc_defaults.alpha);
		pulse[0] = CAST_UP(&ps);
		break;

	case PULSE_SINC_SMS:

		for (int m = 0; m < mb; m++) {

			pm[m] = pulse_sms_defaults;
			pulse_sms_init(&pm[m], dur, flip_angle, 0, bwtp, pulse_sms_defaults.alpha,
				mb, m, sms_dist, slice_th);

			pulse[m] = CAST_UP(&pm[m]);
		}

		break;

	case PULSE_HS:

		pulse_hypsec_init(GYRO, &ph);
		pulse[0] = CAST_UP(&ph);
		break;
	
	case PULSE_REC:

		pulse_rect_init(&pr, dur, flip_angle, 0);
		pulse[0] = CAST_UP(&pr);
		break;
	}

	long dims[DIMS];
	md_singleton_dims(DIMS, dims);
	dims[READ_DIM] = Ntime;
	dims[SLICE_DIM] = mb;

	complex float* signal = create_cfl(out_signal, DIMS, dims);

	for (int m = 0; m < mb; m++)
		for (int t = 0; t < Ntime; t++)
			signal[m * Ntime + t] = pulse_eval(pulse[m], (t + 0.5) * pulse[m]->duration / Ntime);

	unmap_cfl(DIMS, dims, signal);

	return 0;
}


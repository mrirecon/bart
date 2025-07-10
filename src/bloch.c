/* Copyright 2023. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <math.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ode.h"

#include "misc/mri.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "seq/pulse.h"

#include "simu/bloch.h"


#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif


struct sim_data {

	float B0;
	float R1;
	float R2;
	float h;
	float tol;
};

static void simulate(const struct sim_data* data, float out[3], float st, float en, const struct pulse* ps)
{
	NESTED(void, eval, (float out[3], float t, const float in[3]))
	{
		complex float p = pulse_eval(ps, t);
		float gb[3] = { crealf(p), cimagf(p), data->B0 };
		bloch_ode(out, in, data->R1, data->R2, gb);
	};

	ode_interval(data->h, data->tol, 3, out, st, en, eval);
}

static const char help_str[] = "simulation tool";


int main_bloch(int argc, char* argv[argc])
{
	const char* out_signal = NULL;
	const char* out_zmagn = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_signal, "signal"),
		ARG_OUTFILE(true, &out_zmagn, "signal"),
	};

	const struct opt_s opts[] = { };

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);


	long dims[DIMS];
	md_singleton_dims(DIMS, dims);

	dims[SLICE_DIM] = 256;
	dims[READ_DIM] = 256;

	long strs[DIMS];
	md_calc_strides(DIMS, strs, dims, CFL_SIZE);

	complex float* signal = create_cfl(out_signal, DIMS, dims);
	complex float* zmagn = create_cfl(out_zmagn, DIMS, dims);

	long pos[DIMS] = { };

	struct pulse_sinc ps = pulse_sinc_defaults;
	pulse_sinc_init(&ps, 0.001, 90., 0., 4., ps.alpha);

	struct sim_data data = { .B0 = 1., .R1 = 1., .R2 = 50., .h = 1.E-4, .tol = 1.E-6 };
#ifdef  __clang__
	const long *xdims = dims;
	const long *xpos = pos;
	NESTED(double, frac, (int dim)) { return (2. * xpos[dim] - xdims[dim]) / (2. * xdims[dim]); };
#else
	NESTED(float, frac, (int dim)) { return (2. * pos[dim] - dims[dim]) / (2. * dims[dim]); };
#endif
	float dur = CAST_UP(&ps)->duration;
	float delta = dur / dims[READ_DIM];

	float out[3];

	do {
		if (0 == pos[READ_DIM]) {

			out[0] = out[1] = 0.;
			out[2] = 1.;
		}

		data.B0 = 100000. * frac(SLICE_DIM);

		float st = pos[READ_DIM] * delta;
		simulate(&data, out, st, st + delta, CAST_UP(&ps));

		MD_ACCESS(DIMS, strs, pos, signal) = (out[0] + 1.i * out[1]) * cexpf(+1.i * data.B0 * ((st - delta  * (-0/*5*/ + pos[READ_DIM] / 2.)) + delta));
		MD_ACCESS(DIMS, strs, pos, zmagn) = out[2];

	} while (md_next(DIMS, dims, READ_FLAG|SLICE_FLAG, pos));

	unmap_cfl(DIMS, dims, signal);
	unmap_cfl(DIMS, dims, zmagn);

	return 0;
}



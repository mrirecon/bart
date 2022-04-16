/* Copyright 2017-2018. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Siddharth Iyer <ssi@mit.edu>
 *
 * Bilgic B, Gagoski BA, Cauley SF, Fan AP, Polimeni JR, Grant PE, Wald LL, Setsompop K. 
 * Wave‐CAIPI for highly accelerated 3D imaging. Magnetic resonance in medicine. 
 * 2015 Jun 1;73(6):2152-62.
 */

#include <assert.h>
#include <complex.h>
#include <math.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

// Larmor frequency in Hertz per Gauss
#ifndef LARMOR
#define LARMOR 4257.56
#endif

static const char help_str[] = "Generate a wave PSF in hybrid space.\n"
															 "- Assumes the first dimension is the readout dimension.\n"
															 "- Only generates a 2 dimensional PSF.\n"
															 "- Use reshape and fmac to generate a 3D PSF.\n\n"
															 "3D PSF Example:\n"
															 "bart wavepsf		-x 768 -y 128 -r 0.1 -a 3000 -t 0.00001 -g 0.8 -s 17000 -n 6 wY\n"
															 "bart wavepsf -c -x 768 -y 128 -r 0.1 -a 3000 -t 0.00001 -g 0.8 -s 17000 -n 6 wZ\n"
															 "bart reshape 7 wZ 768 1 128 wZ wZ\n"
															 "bart fmac wY wZ wYZ";

int main_wavepsf(int argc, char* argv[argc])
{
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_OUTFILE(true, &out_file, "output"),
	};

	// Spatial dimensions.
	int sx = 512;				// Number of readout points.
	int sy = 128;				// Number of phase encode points.
	float dy = 0.1;			// Resolution in the phase encode direction in cm.

	// ADC parameters.
	int adc = 3000;			// Readout duration in microseconds.
	float dt = 1e-5;		// ADC sampling rate in seconds.

	// Gradient parameters.
	float gmax = 0.8;		// Maximum gradient amplitude in Gauss per centimeter.
	float smax = 17000; // Maximum slew rate in Gauss per centimeter per second.

	// Wave parameters.
	int ncyc = 6;				// Number of gradient sine-cycles.

	// Sine wave or cosine wave.
	bool cs = false;		// Set to true to use a cosine gradient wave/

	const struct opt_s opts[] = {
		OPT_SET(	'c', &cs,							"Set to use a cosine gradient wave"),
		OPT_INT(	'x', &sx,		"RO_dim", "Number of readout points"),
		OPT_INT(	'y', &sy,		"PE_dim", "Number of phase encode points"),
		OPT_FLOAT('r', &dy,		"PE_res", "Resolution of phase encode in cm"),
		OPT_INT(	'a', &adc,	"ADC_T",	"Readout duration in microseconds."),
		OPT_FLOAT('t', &dt,		"ADC_dt", "ADC sampling rate in seconds"),
		OPT_FLOAT('g', &gmax, "gMax",		"Maximum gradient amplitude in Gauss/cm"),
		OPT_FLOAT('s', &smax, "sMax",		"Maximum gradient slew rate in Gauss/cm/second"),
		OPT_INT(	'n', &ncyc, "ncyc",		"Number of cycles in the gradient wave"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(0 == adc % 10);					// Scanners require ADC_duration to be a multiple of 10.

	int wavepoints = adc / 10;				// Number of points in the gradient wave.
	float T = wavepoints * dt / ncyc; // Time period of the sine wave.
	float w = 2 * M_PI / T;					// Frequency in radians per second.

	/* Calculating the wave-amplitude to use. It is either slew limited or gradient 
		 amplitude limited. */
	float gamp = (smax >= w * gmax) ? gmax : smax/w;
	float gwave[wavepoints];

	for (int tdx = 0; tdx < wavepoints; tdx++)
		gwave[tdx] = gamp * ((cs) ? cos(w * tdx * dt) : sin(w * tdx * dt));
	
	complex float phasepercm[wavepoints];
	float prephase = -2 * M_PI * LARMOR * gamp/w;
	float cumsum = 0;

	for (int tdx = 0; tdx < wavepoints; tdx++) {

		phasepercm[tdx] = 2 * M_PI * LARMOR * (cumsum + gwave[tdx] / 2.0) * dt + prephase;
		cumsum = cumsum + gwave[tdx]; 
	}

	// Interpolate to sx via sinc interpolation
	const long wavepoint_dims[1] = {wavepoints};
	const long interp_dims[1] = {sx};

	complex float k_phasepercm[wavepoints]; 

	fftuc(1, wavepoint_dims, 1, k_phasepercm, phasepercm);	

	complex float k_phasepercm_interp[sx]; 

	md_resize_center(1, interp_dims, k_phasepercm_interp, wavepoint_dims, k_phasepercm, 
		sizeof(complex float));

	complex float phasepercm_interp_complex[sx]; 

	ifftuc(1, interp_dims, 1, phasepercm_interp_complex, k_phasepercm_interp);

	complex float phasepercm_interp_real[sx]; 

	md_zreal(1, interp_dims, phasepercm_interp_real, phasepercm_interp_complex);

	complex float phasepercm_interp[sx]; 
	float scale = sqrt((float) sx / wavepoints);

	md_zsmul(1, interp_dims, phasepercm_interp, phasepercm_interp_real, scale);

	complex float psf[sy][sx];

	int midy = sy / 2;

	complex float phase[sx];
	float val;

	for (int ydx = 0; ydx < sy; ydx++) {

		val = -dy * (ydx - midy);

		md_zsmul(1, interp_dims, phase, phasepercm_interp, val);
		md_zexpj(1, interp_dims, psf[ydx], phase);
	}

	const long psf_dims[3] = { sx, sy, 1 };

	complex float* psf_cfl = create_cfl(out_file, 3, psf_dims);

	md_copy(3, psf_dims, psf_cfl, psf, sizeof(complex float));

	unmap_cfl(3, psf_dims, psf_cfl);

	return 0;
}

/* Copyright 2017. Massachusetts Institute of Technology.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Siddharth Iyer <ssi@mit.edu>
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


static const char usage_str[] = "<output>";
static const char help_str[] = "Generate the wave PSF in hybrid space.\n"
															 "- Assumes the first dimension is the readout dimension.\n"
															 "- Assumes Gy and Gz gradients have idential max\n"
															 "	amplitude and slew rate.\n"
                               "Example:\n"
                               "bart wavepsf -x 768 -y 256 -z 1 -p 0.1 -q 0.1 -a 3000 -t 0.00001 -g 4 -s 18000 -n 6 wave_psf\n";

int main_wavepsf(int argc, char* argv[])
{
	
	// Spatial dimensions.
	int sx = 512;				// Number of readout points. Size of dimension 0.
	int sy = 128;				// Number of phase encode 1 points. Size of dimension 1.
	int sz = 1;					// Number of phase encode 2 points. Size of dimension 2.
	float dy = 0.1;			// Resolution in the phase encode 1 direction in cm.
	float dz = 0.1;			// Resolution in the phase encode 2 direction in cm.

	// ADC parameters.
	int adc = 3000;			// Readout duration in microseconds.
	float dt = 1e-5;		// ADC sampling rate in seconds.

	// Gradient parameters.
	float gmax = 4;			// Maximum gradient amplitude in Gauss per centimeter.
	float smax = 18000; // Maximum slew rate in Gauss per centimeter per second.

	// Wave parameters.
	int ncyc = 6;				// Number of gradient sine-cycles.

	const struct opt_s opts[] = {
		OPT_INT('x', &sx, "DIM_ro", "Number of readout points or numel(dim 0)"),
		OPT_INT('y', &sy, "DIM_pe1", "Number of phase encode 1 points or numel(dim 1)"),
		OPT_INT('z', &sz, "DIM_pe2", "Number of phase encode 2 points or numel(dim 2)"),
		OPT_FLOAT('p', &dy, "RES_pe1", "Resolution in phase encode 1 (centimeters)"),
		OPT_FLOAT('q', &dz, "RES_pe2", "Resolution in phase encode 2 (centimeters)"),
		OPT_INT('a', &adc, "ADC_duration", "Readout duration in microseconds."),
		OPT_FLOAT('t', &dt, "ADC_dt", "ADC sampling rate in seconds"),
		OPT_FLOAT('g', &gmax, "GRAD_maxamp", "Maximum gradient amplitude in Gauss/cm"),
		OPT_FLOAT('s', &smax, "GRAD_maxslew", "Maximum gradient slew rate in Gauss/cm/second"),
		OPT_INT('n', &ncyc, "WAVE_cycles", "Number of cycles in the gradient sine wave."),
	};

	cmdline(&argc, argv, 1, 1, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	assert(0 == adc % 10);					// Scanners require ADC_duration to be a multiple of 10.

	int wavepoints = adc/10;				// Number of points in the gradient wave.
	float T = wavepoints * dt/ncyc; // Time period of the sine wave.
	float w = 2 * M_PI/T;						// Frequency in radians per second.

	/* Calculating the sine-amplitude to use. It is ether slew limited or gradient 
		 amplitude limited. */
	float gamp = (smax >= w * gmax) ? gmax : smax/w;
	float gwave[wavepoints];
	for (int tdx = 0; tdx < wavepoints; tdx++) {
		gwave[tdx] = gamp * sin(w * tdx * dt);
	}
	
	complex float phasepercm[wavepoints];
	float prephase = -2 * M_PI * LARMOR * gamp/w;
	float cumsum = 0;
	for (int tdx = 0; tdx < wavepoints; tdx++) {
		phasepercm[tdx] = 2 * M_PI * LARMOR * (cumsum + gwave[tdx]/2.0) * dt + prephase;
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
	float scale = sqrt((float) sx/wavepoints);
	md_zsmul(1, interp_dims, phasepercm_interp, phasepercm_interp_real, scale);

	complex float psf[sz][sy][sx]; //Dimensions reversed to be consistent with cfl

	int midy = sy/2;
	int midz = sz/2;

	complex float phase[sx];
	float val;

	for (int ydx = 0; ydx < sy; ydx++) {
		for (int zdx = 0; zdx < sz; zdx++) {
			val = -((ydx - midy) * dy + (zdx - midz) * dz);
			md_zsmul(1, interp_dims, phase, phasepercm_interp, val);
			md_zexpj(1, interp_dims, psf[zdx][ydx], phase);
		}
	}

	const long psf_dims[3] = {sx, sy, sz};
	complex float* psf_cfl = create_cfl(argv[1], 3, psf_dims);
	md_copy(3, psf_dims, psf_cfl, psf, sizeof(complex float));
	unmap_cfl(3, psf_dims, psf_cfl);

	exit(0);
}

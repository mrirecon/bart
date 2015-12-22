/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/fft.h"

#include "misc/mmio.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char* usage_str = "[-u] [-i] bitmask <input> <output>";
static const char* help_str =
		"Performs a fast Fourier transform (FFT) along selected dimensions.\n";




int main_fft(int argc, char* argv[])
{
	bool unitary = false;
	bool inv = false;

	cmdline(argc, argv, 3, usage_str, help_str, 2,
		(struct opt_s[2]){	{ 'u', false, opt_set, &unitary, "unitary" },
					{ 'i', false, opt_set, &inv, "inverse" },	});


	long dims[DIMS];
	complex float* idata = load_cfl(argv[2], DIMS, dims);
	complex float* data = create_cfl(argv[3], DIMS, dims);

	unsigned long flags = labs(atol(argv[1]));


	md_copy(DIMS, dims, data, idata, sizeof(complex float));
	unmap_cfl(DIMS, dims, idata);

	if (unitary)
		fftscale(DIMS, dims, flags, data, data);

	(inv ? ifftc : fftc)(DIMS, dims, flags, data, data);

	unmap_cfl(DIMS, dims, data);
	exit(0);
}



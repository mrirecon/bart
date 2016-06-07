/* Copyright 2013. The Regents of the University of California.
 * Copyright 2015-2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>

#include "num/multind.h"
#include "num/fft.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/opts.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "bitmask <input> <output>";
static const char help_str[] = "Performs a fast Fourier transform (FFT) along selected dimensions.";




int main_fft(int argc, char* argv[])
{
	bool unitary = false;
	bool inv = false;

	const struct opt_s opts[] = {

		OPT_SET('u', &unitary, "unitary"),
		OPT_SET('i', &inv, "inverse"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

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



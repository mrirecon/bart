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


static const char help_str[] = "Performs a fast Fourier transform (FFT) along selected dimensions.";




int main_fft(int argc, char* argv[argc])
{
	unsigned long flags = 0;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_ULONG(true, &flags, "bitmask"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	bool unitary = false;
	bool inv = false;
	bool center = true;

	const struct opt_s opts[] = {

		OPT_SET('u', &unitary, "unitary"),
		OPT_SET('i', &inv, "inverse"),
		OPT_CLEAR('n', &center, "un-centered"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	complex float* idata = load_cfl(in_file, DIMS, dims);
	complex float* data = create_cfl(out_file, DIMS, dims);


	md_copy(DIMS, dims, data, idata, sizeof(complex float));
	unmap_cfl(DIMS, dims, idata);

	if (unitary)
		fftscale(DIMS, dims, flags, data, data);

	(inv ? (center ? ifftc : ifft) : (center ? fftc : fft))(DIMS, dims, flags, data, data);

	unmap_cfl(DIMS, dims, data);

	return 0;
}



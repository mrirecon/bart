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

#include "num/init.h"

#include "na/na.h"
#include "na/io.h"
#include "na/math.h"

#include "misc/opts.h"
#include "misc/misc.h"



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

	na in = na_load(argv[2]);
	na out = na_create(argv[3], na_type(in));

	unsigned long flags = labs(atol(argv[1]));

	na_copy(out, in);
	na_free(in);

	__typeof__(na_fft)* ffts[2][2] = {
		{ na_fftc, na_ifftc },
		{ na_fftuc, na_ifftuc },
	};

	ffts[unitary][inv](flags, out, out);

	na_free(out);
	exit(0);
}



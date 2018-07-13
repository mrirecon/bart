/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/filter.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"


#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "flags <input> <output>";
static const char help_str[] = "Apply Hamming (Hann) window to <input> along dimensions specified by flags";


int main_window(int argc, char* argv[])
{
	bool hamming = true;

	const struct opt_s opts[] = {

		OPT_CLEAR('H', &hamming, "Hann window"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long dims[DIMS];
	
	long flags = atoi(argv[1]);
	complex float* in_data = load_cfl(argv[2], DIMS, dims);
	complex float* out_data = create_cfl(argv[3], DIMS, dims);

	(hamming ? md_zhamming : md_zhann)(DIMS, dims, flags, out_data, in_data);

	unmap_cfl(DIMS, dims, in_data);
	unmap_cfl(DIMS, dims, out_data);
	return 0;
}



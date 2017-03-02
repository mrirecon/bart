/* Copyright 2016-2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2017	Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <complex.h>
#include <stdbool.h>
#include <stdio.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"

#include "calib/cc.h"




static const char usage_str[] = "<kspace> <cc_matrix> <proj_kspace>";
static const char help_str[] = "Apply coil compression forward/inverse operation.";



int main_ccapply(int argc, char* argv[])
{
	bool forward = true;
	bool do_fft = true;
	long P = -1;
	enum cc_type { SCC, GCC, ECC } cc_type = SCC;

	const struct opt_s opts[] = {

		OPT_LONG('p', &P, "N", "perform compression to N virtual channels"),
		OPT_CLEAR('u', &forward, "apply inverse operation"),
		OPT_CLEAR('t', &do_fft, "don't apply FFT in readout"),
		OPT_SELECT('S', enum cc_type, &cc_type, SCC, "type: SVD"),
		OPT_SELECT('G', enum cc_type, &cc_type, GCC, "type: Geometric"),
		OPT_SELECT('E', enum cc_type, &cc_type, ECC, "type: ESPIRiT"),
	};

	cmdline(&argc, argv, 3, 3, usage_str, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	long in_dims[DIMS];
	long cc_dims[DIMS];

	complex float* in_data = load_cfl(argv[1], DIMS, in_dims);
	complex float* cc_data = load_cfl(argv[2], DIMS, cc_dims);

	assert(1 == in_dims[MAPS_DIM]);
	const long channels = cc_dims[COIL_DIM];

	if (-1 == P)
		P = in_dims[COIL_DIM];

	assert(cc_dims[MAPS_DIM] >= P && in_dims[COIL_DIM] >= P);

	long out_dims[DIMS] = MD_INIT_ARRAY(DIMS, 1);

	md_select_dims(DIMS, ~COIL_FLAG, out_dims, in_dims);
	out_dims[COIL_DIM] = forward ? P : channels;
	
	complex float* out_data = create_cfl(argv[3], DIMS, out_dims);

	// transpose for the matrix multiplication
	long trp_dims[DIMS];

	if (forward) {

		debug_printf(DP_DEBUG1, "Compressing to %ld virtual coils...\n", P);

		md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, trp_dims, out_dims);
		trp_dims[MAPS_DIM] = out_dims[COIL_DIM];

	} else {

		debug_printf(DP_DEBUG1, "Uncompressing channels...\n");

		md_transpose_dims(DIMS, COIL_DIM, MAPS_DIM, trp_dims, in_dims);
	}

	long cc2_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cc2_dims, cc_dims);
	cc2_dims[MAPS_DIM] = P;

	if (SCC != cc_type) {

		if (do_fft) {

			complex float* in2_data = anon_cfl(NULL, DIMS, in_dims);

			ifftuc(DIMS, in_dims, READ_FLAG, in2_data, in_data);

			unmap_cfl(DIMS, in_dims, in_data);
			in_data = in2_data;
		}

		complex float* cc2_data = anon_cfl(NULL, DIMS, cc2_dims);
		align_ro(cc2_dims, cc2_data, cc_data);

		unmap_cfl(DIMS, cc_dims, cc_data);
		cc_data = cc2_data;
	}

	if (forward)
		md_zmatmulc(DIMS, trp_dims, out_data, cc2_dims, cc_data, in_dims, in_data);
	else
		md_zmatmul(DIMS, out_dims, out_data, cc2_dims, cc_data, trp_dims, in_data);

	if (SCC != cc_type) {

		if (do_fft)
			fftuc(DIMS, out_dims, READ_FLAG, out_data, out_data);

		unmap_cfl(DIMS, cc2_dims, cc_data);

	} else {

		unmap_cfl(DIMS, cc_dims, cc_data);
	}

	unmap_cfl(DIMS, in_dims, in_data);
	unmap_cfl(DIMS, out_dims, out_data);

	printf("Done.\n");

	exit(0);
}



/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/misc.h" 
#include "misc/debug.h" 
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/opts.h"

#include "ismrm/read.h"


static const char help_str[] = "Import ISMRM raw data files.";


int main_ismrmrd(int argc, char* argv[argc])
{
	const char* ismrm_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_STRING(true, &ismrm_file, "input"),
		ARG_OUTFILE(false, &out_file, "output"),
	};

	struct isrmrm_config_s config = ismrm_default_config;

	const struct opt_s opts[] = {
		
		OPTL_INT('m', "measurement", &(config.measurement), "", "select specific measurement (split by flag)"),
		OPTL_INT('o', "overwrite", &(config.overwriting_idx), "idx", "overwrite data idx times"),

		//OPTL_SELECT(0, "interleaved",enum ISMRMRD_SLICE_ORDERING, &(config.slice_ord),ISMRMRD_SLICE_INTERLEAVED, "interleaved slice ordering (1, 3, 5, 2, 4) / (1, 3, 2, 4)"),
		OPTL_SELECT(0, "interleaved-siemens",enum ISMRMRD_SLICE_ORDERING, &(config.slice_ord),ISMRMRD_SLICE_INTERLEAVED_SIEMENS, "interleaved slice ordering (1, 3, 5, 2, 4) / (2, 4, 1, 3)"),

	};
	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (NULL == out_file) {

		ismrm_print_xml(ismrm_file);
		return 0;
	}

	long dims[DIMS];

	debug_printf(DP_INFO, "Reading dims ...\n");

	ismrm_read_dims(ismrm_file, &config, DIMS, dims);

	debug_printf(DP_INFO, "done.\nDIMS: ");
	debug_print_dims(DP_INFO, DIMS, dims);
	

	complex float* out = create_cfl(out_file, DIMS, dims);
	md_clear(DIMS, dims, out, CFL_SIZE);

	debug_printf(DP_INFO, "Reading data ...\n");

	ismrm_read(ismrm_file, &config, DIMS, dims, out);

	debug_printf(DP_INFO, "done.\n");

	unmap_cfl(DIMS, dims, out);

	return 0;
}





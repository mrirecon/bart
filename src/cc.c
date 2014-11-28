/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2014	Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013		Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"

#include "calib/cc.h"




static void usage(const char* name, FILE* fd)
{
	fprintf(fd, 	"Usage: %s [-A] [-r cal_size] [-P num_coeffs]"
			" <kspace> <coeff>|<proj_kspace>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Performs coil compression.\n"
		"\n"
		"-P N\tperform compression to N virtual channels\n"
		"-r S\tsize of calibration region\n"
		"-A\tuse all data to compute coefficients\n"
		"-S|G|E\ttype: SVD, Geometric, ESPIRiT\n"
		"-h\thelp\n");
}




int main_cc(int argc, char* argv[])
{
	long calsize[3] = { 24, 24, 24 };
	bool proj = false;
	long P = 0;
	bool all = false;
	enum cc_type { SCC, GCC, ECC } cc_type = SCC;

	int c;
	while (-1 != (c = getopt(argc, argv, "r:R:SGEP:Ah"))) {

		switch (c) {

		case 'A':
			all = true;
			break;

		case 'r':
			calsize[0] = atoi(optarg);
			calsize[1] = atoi(optarg);
			calsize[2] = atoi(optarg);
			break;

		case 'R':
			sscanf(optarg, "%ld:%ld:%ld", &calsize[0], &calsize[1], &calsize[2]);
			break;

		case 'P':
			proj = true;
			P = atoi(optarg);
			break;

		case 'S':
			cc_type = SCC;
			break;

		case 'G':
			cc_type = GCC;
			break;

		case 'E':
			cc_type = ECC;
			break;

		case 'h':
			usage(argv[0], stdout);
			help();
			exit(0);

		default:
			usage(argv[0], stderr);
			exit(1);
		}
	}

	if (argc - optind != 2) {

		fprintf(stderr,"Input arguments do not match expected format.\n");
		usage(argv[0], stderr);
		exit(1);
	}


	long in_dims[DIMS];

	complex float* in_data = load_cfl(argv[optind + 0], DIMS, in_dims);

	assert(1 == in_dims[MAPS_DIM]);
	long channels = in_dims[COIL_DIM];

	if (0 == P)
		P = channels;

	long out_dims[DIMS] = MD_INIT_ARRAY(DIMS, 1);
	out_dims[COIL_DIM] = channels;
	out_dims[MAPS_DIM] = channels;
	out_dims[READ_DIM] = (SCC == cc_type) ? 1 : in_dims[READ_DIM];

	complex float* out_data = (proj ? anon_cfl : create_cfl)(argv[optind + 1], DIMS, out_dims);


	long caldims[DIMS];
	complex float* cal_data = NULL;

	if (all) {

		md_copy_dims(DIMS, caldims, in_dims);
		cal_data = in_data;

	} else {
		
		cal_data = extract_calib(caldims, calsize, in_dims, in_data, false);
	}

	if (ECC == cc_type)
		debug_printf(DP_WARN, "Warning: ECC depends on a parameter choice rule for optimal results which is not implemented.\n");


	switch (cc_type) {
	case SCC: scc(out_dims, out_data, caldims, cal_data); break;
	case GCC: gcc(out_dims, out_data, caldims, cal_data); break;
	case ECC: ecc(out_dims, out_data, caldims, cal_data); break;
	}

	if (!all)
		md_free(cal_data);


	if (proj) {

		debug_printf(DP_DEBUG1, "Compressing to %ld virtual coils...\n", P);

		long trans_dims[DIMS];
		md_copy_dims(DIMS, trans_dims, in_dims);
		trans_dims[COIL_DIM] = P;

		complex float* trans_data = create_cfl(argv[optind + 1], DIMS, trans_dims);

		long fake_trans_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, fake_trans_dims, in_dims);
		fake_trans_dims[MAPS_DIM] = P;

		long out2_dims[DIMS];
		md_copy_dims(DIMS, out2_dims, out_dims);
		out2_dims[MAPS_DIM] = P;

		if (SCC != cc_type) {

			complex float* in2_data = anon_cfl(NULL, DIMS, in_dims);

			ifftuc(DIMS, in_dims, READ_FLAG, in2_data, in_data);

			unmap_cfl(DIMS, in_dims, in_data);
			in_data = in2_data;


			complex float* out2 = anon_cfl(NULL, DIMS, out2_dims);
			align_ro(out2_dims, out2, out_data);

			unmap_cfl(DIMS, out_dims, out_data);
			out_data = out2;
		}

		md_zmatmulc(DIMS, fake_trans_dims, trans_data, out2_dims, out_data, in_dims, in_data);

		if (SCC != cc_type) {

			fftuc(DIMS, trans_dims, READ_FLAG, trans_data, trans_data);

			unmap_cfl(DIMS, out2_dims, out_data);

		} else {

			unmap_cfl(DIMS, out_dims, out_data);
		}

		unmap_cfl(DIMS, trans_dims, trans_data);
		unmap_cfl(DIMS, in_dims, in_data);

	} else {

		unmap_cfl(DIMS, in_dims, in_data);
		unmap_cfl(DIMS, out_dims, out_data);
	}

	printf("Done.\n");
	exit(0);
}



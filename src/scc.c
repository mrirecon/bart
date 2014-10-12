/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013	Martin Uecker <uecker@eecs.berkeley.edu>
 * 2013		Jonathan Tamir <jtamir@eecs.berkeley.edu>
 * 
 *
 * Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR. A software channel
 * compression technique for faster reconstruction with many channels.
 * Magn Reson Imaging 2008; 26:133-141.
 *
 * Buehrer M, Pruessmann KP, Boesiger P, Kozerke S. Array compression for MRI
 * with large coil arrays. Magn Reson Med 2007, 57: 1131–1139. 
 * 
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>

#include "misc/io.h"
#include "misc/mmio.h"
#include "misc/mri.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/lapack.h"
#include "num/la.h"




static void usage(const char* name, FILE* fd)
{
	fprintf(fd, 	"Usage: %s [-v] [-A] [-r cal_size] [-P num_coeffs]"
			" <kspace> <coeff>|<proj_kspace>\n", name);
}

static void help(void)
{
	printf( "\n"
		"Performs simple coil compression.\n"
		"\n"
		"-P N\tperform compression to N virtual channels\n"
		"-r S\tsize of calibration region\n"
		"-A\tuse all data to compute coefficients\n"
		"-v\tverbose\n"
		"-h\thelp\n");
}




int main_scc(int argc, char* argv[])
{
	long calsize[3] = { 24, 24, 24 }; 
	bool proj = false;
	bool verbose = false;
	long P = 0;
	bool all = false;

	int c;
	while (-1 != (c = getopt(argc, argv, "r:R:p:P:vAh"))) {

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

		case 'v':
			verbose = true;
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

	complex float* out_data = NULL;
	
	if (proj)
		out_data = md_alloc(DIMS, out_dims, CFL_SIZE);
	else
		out_data = create_cfl(argv[optind + 1], DIMS, out_dims);


	long caldims[DIMS];
	complex float* cal_data = NULL;

	if (all) {

		md_copy_dims(DIMS, caldims, in_dims);
		cal_data = in_data;

	} else {
		
		cal_data = extract_calib(caldims, calsize, in_dims, in_data, false);
	}

	

	complex float* tmp = xmalloc(channels * channels * CFL_SIZE);
	size_t csize = md_calc_size(3, caldims);
	gram_matrix(channels, (complex float (*)[channels])tmp, csize, (const complex float (*)[csize])cal_data);

	float vals[channels];
	eigendecomp(channels, vals, (complex float (*)[])tmp);
	md_flip(DIMS, out_dims, MAPS_FLAG, out_data, tmp, CFL_SIZE);

	free(tmp);

	if (verbose) {

		printf("Coefficients:");

		for (int i = 0; i < channels; i++)
			printf(" %.3f", vals[channels - 1 - i] / vals[channels - 1]);

		printf("\n");
	}



	if (proj) {
	
		if (verbose)
			printf("Compressing to %ld virtual coils...\n", P);

		long trans_dims[DIMS];
		md_copy_dims(DIMS, trans_dims, in_dims);
		trans_dims[COIL_DIM] = P;

		complex float* trans_data = create_cfl(argv[optind + 1], DIMS, trans_dims);

		long fake_trans_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, fake_trans_dims, in_dims);
		fake_trans_dims[MAPS_DIM] = P;

		md_zmatmulc(DIMS, fake_trans_dims, trans_data, out_dims, out_data, in_dims, in_data);

		unmap_cfl(DIMS, trans_dims, trans_data);
	}

	printf("Done.\n");

	if (!all)
		md_free(cal_data);

	unmap_cfl(DIMS, in_dims, in_data);

	if (proj)
		md_free(out_data);
	else
		unmap_cfl(DIMS, out_dims, out_data);

	exit(0);
}



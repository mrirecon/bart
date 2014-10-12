/* Copyright 2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#define _GNU_SOURCE
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <getopt.h>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/mmio.h"

#include "simu/phantom.h"



static void usage(const char* name, FILE* fd)
{
	fprintf(fd, "Usage: %s [-k | -s nc] [-t trajectory] <output>\n", name);
}


static void help(void)
{
	printf( "\n"
		"Image and k-space domain phantoms.\n"
		"\n"
		"-s nc\tnc sensitivities\n"
		"-k\tk-space\n"
		"-h\thelp\n");
}



int main_phantom(int argc, char* argv[])
{
	int c;
	_Bool kspace = false;
	int sens = 0;
	_Bool out_sens = false;
	_Bool circ = false;
	char* traj = NULL;

	long dims[DIMS] = { [0 ... DIMS - 1] = 1 };
	dims[0] = 1;
	dims[1] = 128;
	dims[2] = 128;

	while (-1 != (c = getopt(argc, argv, "x:kcS:s:t:h"))) {

		switch (c) {

		case 'x':
			dims[1] = dims[2] = atoi(optarg);
			break;

		case 'k':
			kspace = true;
			break;

		case 'S':
			out_sens = true;
		case 's':
			sens = atoi(optarg);
			break;

		case 'c':
			circ = true;
			break;

		case 't':
			traj = strdup(optarg);
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

	if (argc - optind != 1) {
		
		usage(argv[0], stderr);
		exit(1);
	}


	long sdims[DIMS];
	complex float* samples = NULL;

	if (NULL != traj) {

		samples = load_cfl(traj, DIMS, sdims);

		dims[0] = 1;
		dims[1] = sdims[1];
		dims[2] = sdims[2];
	}


	if (sens)
		dims[3] = sens;

	complex float* out = create_cfl(argv[optind + 0], DIMS, dims);

	if (out_sens) {

		assert(NULL == traj);
		assert(!kspace);

		calc_sens(dims, out);

	} else
	if (circ) {

		assert(NULL == traj);

		calc_circ(dims, out, kspace);
//		calc_ring(dims, out, kspace);

	} else {

		//assert(1 == dims[COIL_DIM]);

		if (NULL == samples) {

			calc_phantom(dims, out, kspace);

		} else {

			dims[0] = 3;
			calc_phantom_noncart(dims, out, samples);
			dims[0] = 1;
		}
	}

	if (NULL != traj)
		free(traj);

	if (NULL != samples)
		unmap_cfl(3, sdims, samples);

	unmap_cfl(DIMS, dims, out);
	exit(0);
}



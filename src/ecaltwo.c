/* Copyright 2013. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012-2013 Martin Uecker <uecker@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <stdio.h>
#include <getopt.h>

#include "num/multind.h"
#include "calib/calib.h"
#include "num/fft.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"



static void usage(const char* name, FILE* fp)
{
	fprintf(fp, "Usage: %s [-c crop] [-m maps] x y z <input> <sensitivities> [<ev_maps>]\n", name);
}


static void help(void)
{
	printf( "\n"
		"Second part of ESPIRiT calibration.\n"
		"Optionally outputs the eigenvalue maps.\n"
		"\n"
		"-c crop_value\tCrop the sensitivities if the eigenvalue is smaller than {crop_value}.\n"
		"-m maps\t\tNumber of maps to compute.\n");
}



int main(int argc, char* argv[])
{
	long maps = 2; // channels;
	struct ecalib_conf conf;
	memcpy(&conf, &ecalib_defaults, sizeof(struct ecalib_conf));

	int c;
	while (-1 != (c = getopt(argc, argv, "OSc:m:gh"))) {

		switch (c) {

		case 'S':
			conf.softcrop = true;
			break;

		case 'O':
			conf.orthiter = false;
			break;

		case 'c':
			conf.crop = atof(optarg);
			break;

		case 'm':
			maps = atoi(optarg);
			break;

		case 'g':
			conf.usegpu = true;
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

	if ((argc - optind != 5) && (argc - optind != 6)) {

		usage(argv[0], stderr);
		exit(1);
	}

	long in_dims[KSPACE_DIMS];

	complex float* in_data = load_cfl(argv[optind + 3], KSPACE_DIMS, in_dims);

	int channels = 0;

	while (in_dims[3] != (channels * (channels + 1) / 2))
		channels++;

	printf("Channels: %d\n", channels);

	assert(maps <= channels);


	long out_dims[KSPACE_DIMS] = { [0 ... KSPACE_DIMS - 1] = 1 };
	long map_dims[KSPACE_DIMS] = { [0 ... KSPACE_DIMS - 1] = 1 };
	
	out_dims[0] = atoi(argv[optind + 0]);
	out_dims[1] = atoi(argv[optind + 1]);
	out_dims[2] = atoi(argv[optind + 2]);
	out_dims[3] = channels;
	out_dims[4] = maps;

	assert((out_dims[0] >= in_dims[0]));
	assert((out_dims[1] >= in_dims[1]));
	assert((out_dims[2] >= in_dims[2]));


	for (int i = 0; i < 3; i++)
		map_dims[i] = out_dims[i];

	map_dims[3] = 1;
	map_dims[4] = maps;


	complex float* out_data = create_cfl(argv[optind + 4], KSPACE_DIMS, out_dims);
	complex float* emaps;

	if (6 == argc - optind)
		emaps = create_cfl(argv[optind + 5], KSPACE_DIMS, map_dims);
	else
		emaps = md_alloc(KSPACE_DIMS, map_dims, sizeof(complex float));

	caltwo(&conf, out_dims, out_data, emaps, in_dims, in_data, NULL, NULL);

	printf("Done.\n");

	unmap_cfl(KSPACE_DIMS, in_dims, in_data);
	unmap_cfl(KSPACE_DIMS, out_dims, out_data);

	if (6 == argc - optind)
		unmap_cfl(KSPACE_DIMS, map_dims, emaps);
	else
		md_free(emaps);

	exit(0);
}



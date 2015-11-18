/* Copyright 2015. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2015 Siddharth Iyer <sid8795@gmail.com>
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

#include "num/flpmath.h"
#include "num/multind.h"

#include "misc/debug.h"

#include "calib/estvar.h"


static void usage(const char* name, FILE* fd)
{
    fprintf(fd, "Usage: %s [-k kernel_size] [-r cal_size] <kspace> \n", name);
}


static void help(void)
{
    printf( "\n"
        "Estimate the noise variance assuming white Gaussian noise."
        "\n"
        "-k ksize\tkernel size\n"
        "-r cal_size\tLimits the size of the calibration region.\n");
}


int main_estvar(int argc, char* argv[])
{

    long calsize_dims[3]  = { 24, 24, 24};
    long kernel_dims[3]   = {  6,  6,  6};

    int c;
    while (-1 != (c = getopt(argc, argv, "k:r:h"))) {

        switch (c) {

            case 'k':
                kernel_dims[0] = atoi(optarg);
                kernel_dims[1] = atoi(optarg);
                kernel_dims[2] = atoi(optarg);
                break;

            case 'r':
                calsize_dims[0] = atoi(optarg);
                calsize_dims[1] = atoi(optarg);
                calsize_dims[2] = atoi(optarg);
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

    int  N = DIMS;
    long kspace_dims[N];

    complex float* kspace = load_cfl(argv[optind + 0], N, kspace_dims);

    for (int idx=0;idx < 3; idx++) {
        kernel_dims[idx]  = (kspace_dims[idx] == 1)? 1 : kernel_dims[idx];
        calsize_dims[idx] = (kspace_dims[idx] == 1)? 1 : calsize_dims[idx];
    }

    float variance = estvar_kspace(N, kernel_dims, calsize_dims, kspace_dims, kspace);

    unmap_cfl(N, kspace_dims, kspace);

    printf("Estimated noise variance: %f\n", variance);

    exit(0);

}

/* Copyright 2018. Damien Nguyen
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <string.h>

#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"

#include "misc/mmio.h"
#include "misc/misc.h"

#ifndef DIMS
#define DIMS 16
#endif

static const char usage_str[] = "<input> <output>";
static const char help_str[] = "Copy data from a file to another";


int main_fcopy(int argc, char* argv[])
{
     mini_cmdline(&argc, argv, 2, usage_str, help_str);

     num_init();
     
     long dims[DIMS];

     complex float* idata = load_cfl(argv[1], DIMS, dims);
     complex float* odata = create_cfl(argv[2], DIMS, dims);

     if (idata == NULL) {
	  BART_ERR("in_data == NULL!");
	  return 1;
     }
     if (odata == NULL) {
	  BART_ERR("out_data == NULL!");
	  return 1;
     }

     memcpy(odata, idata, md_calc_size(DIMS, dims) * sizeof(complex float));

     unmap_cfl(DIMS, dims, idata);
     unmap_cfl(DIMS, dims, odata);
     
     return 0;
}



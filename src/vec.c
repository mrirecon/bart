/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <string.h>

#include "num/multind.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/io.h"
#include "misc/misc.h"


static const char usage_str[] = "val1 val2 ... valN name";
static const char help_str[] = "Create a vector of values.\n";



int main_vec(int argc, char* argv[])
{
	mini_cmdline(&argc, argv, -1, usage_str, help_str);

	num_init();

	long dims[1] = { argc - 2 };

	complex float* x = create_cfl(argv[argc - 1], 1, dims);

	for (int i = 0; i < argc - 2; i++)
		if (0 != parse_cfl(&x[i], argv[1 + i]))
			error("argument %d/%d is not a number: %s", i, argc - 2, argv[1 + i]);

	unmap_cfl(1, dims, x);
	return 0;
}



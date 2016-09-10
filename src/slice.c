/* Copyright 2013. The Regents of the University of California.
 * Copyright 2016. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2012, 2016 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <complex.h>

#include "num/init.h"
#include "num/multind.h" // MD_BIT

#include "na/na.h"
#include "na/io.h"

#include "misc/misc.h"



static const char usage_str[] = "dimension position <input> <output>";
static const char help_str[] = "Extracts a slice from {position} along {dimension}.\n";


int main_slice(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 4, usage_str, help_str);

	num_init();

	na in = na_load(argv[3]);

	int dim = atoi(argv[1]);
	int pos = atoi(argv[2]);

	assert(dim >= 0);
	assert(pos >= 0);
	assert((unsigned int)dim < na_rank(in));
	assert((unsigned int)pos < (*NA_DIMS(in))[dim]);

	long posv[na_rank(in)];

	for (unsigned int i = 0; i < na_rank(in); i++)
		posv[i] = 0;

	posv[dim] = pos;

	na sl = na_slice(in, ~MD_BIT(dim), na_rank(in), &posv);
	na out = na_create(argv[4], na_type(sl));
	na_copy(out, sl);

	na_free(sl);
	na_free(out);
	na_free(in);
	exit(0);
}



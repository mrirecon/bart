/* Copyright 2015-2016. Martin Uecker
 * Copyright 2016. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors: 
 * 2015-2-16 Martin Uecker <uecker@med.uni-goettingen.de>
 * 2016 Jonathan Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/version.h"


static const char usage_str[] = "[-h]";
static const char help_str[] = 
	"Print BART version. The version string is of the form\n"
	"TAG or TAG-COMMITS-SHA as produced by 'git describe'. It\n"
	"specifies the last release (TAG), and (if git is used)\n"
	"the number of commits (COMMITS) since this release and\n"
	"the abbreviated hash of the last commit (SHA). If there\n"
	"are local changes '-dirty' is added at the end.\n";
			

int main_version(int argc, char* argv[])
{
	bool verbose = false;

	const struct opt_s opts[] = {

		OPT_SET('V', &verbose, "Output verbose info"),
	};

	cmdline(&argc, argv, 0, 0, usage_str, help_str, ARRAY_SIZE(opts), opts);

	printf("%s\n", bart_version);

	if (verbose) {

#ifdef __GNUC__
		printf("GCC_VERSION=%s\n", __VERSION__);
#endif

		printf("CUDA=");
#ifdef USE_CUDA
			printf("1\n");
#else
			printf("0\n");
#endif

		printf("ACML=");
#ifdef USE_ACML
			printf("1\n");
#else
			printf("0\n");
#endif

		printf("FFTWTHREADS=");
#ifdef FFTWTHREADS
			printf("1\n");
#else
			printf("0\n");
#endif
	}

	exit(0);
}




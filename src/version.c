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
			
#ifdef USE_CUDA
#  define USE_CUDA_NUM 1
#else
#  define USE_CUDA_NUM 0
#endif /* USE_CUDA */

#ifdef USE_ACML
#  define USE_ACML_NUM 1
#else
#  define USE_ACML_NUM 0
#endif /* USE_ACML */

#ifdef FFTWTHREADS
#  define FFTWTHREADS_NUM 1
#else
#  define FFTWTHREADS_NUM 0
#endif /* FFTWTHREADS */


int main_version(int argc, char* argv[])
{
	return in_mem_version_main(argc, argv, NULL);
}

int in_mem_version_main(int argc, char* argv[], char* output)
{
	int idx = 0;
	int max_length = 512;
	bool verbose = false;

	const struct opt_s opts[] = {

		OPT_SET('V', &verbose, "Output verbose info"),
	};

	cmdline(&argc, argv, 0, 0, usage_str, help_str, ARRAY_SIZE(opts), opts);

	if (output != NULL) {
		idx += safeneg_snprintf(output+idx, max_length-idx, "%s", bart_version);
	}
	else {
		printf("%s\n", bart_version);
	}

	if (verbose) {

#ifdef __GNUC__
		if (output != NULL) {
			idx += safeneg_snprintf(output+idx, max_length-idx, "\nGCC_VERSION=%s", __VERSION__);
		}
		else {
			printf("GCC_VERSION=%s\n", __VERSION__);
		}
#endif

		printf("CUDA=");
		if (output != NULL) {
			idx += safeneg_snprintf(output+idx, max_length-idx, "\nCUDA=%d", USE_CUDA_NUM);
		}
		else {
			printf("CUDA=%d\n", USE_CUDA_NUM);
		}

		if (output != NULL) {
			idx += safeneg_snprintf(output+idx, max_length-idx, "\nACML=%d", USE_ACML_NUM);
		}
		else {
			printf("ACML=%d\n", USE_ACML_NUM);
		}

		if (output != NULL) {
			idx += safeneg_snprintf(output+idx, max_length-idx, "\nFFTWTHREADS=%d", FFTWTHREADS_NUM);
		}
		else {
			printf("FFTWTHREADS=%d\n", FFTWTHREADS_NUM);
		}
	}

	return 0;
}




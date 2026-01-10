/* Copyright 2023-2025. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2023 Nick Scholand <scholand@tugraz.at>
 */

#include <stdbool.h>
#include <complex.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/cppmap.h"

#ifndef DIMS
#define DIMS 16
#endif

#define FUNC_LIST zabs, zacosr, zarg, zatanr, zconj, zcos, zcosh, zexp, zimag, zlog, zphsr, zreal, zround, zsin, zsinh, zsqrt, zsetnanzero, ()

static const char help_str[] = "Perform function evaluation on array.";


typedef void (*function)(int D, const long dims[D], complex float* optr, const complex float* iptr);

struct {

	function func;
	const char* name;

} calc_table[] = {

#define DENTRY(x) { md_ ## x, # x },
	MAP(DENTRY, FUNC_LIST)
#undef  DENTRY
	{ NULL, NULL }
};

static bool help_func_calc(void* /*ptr*/, char /*c*/, const char* /*optarg*/)
{
	printf( "Available functions are:\n");

	for (int i = 0; i < (int)ARRAY_SIZE(calc_table); i++) {

		if (0 == i % 6)
			printf("\n");

		if (NULL != calc_table[i].name)
			printf("%s\t", calc_table[i].name);
	}

	printf("\n");

	exit(0);
}

int main_calc(int argc, char* argv[argc])
{
	const char* in_file = NULL;
	const char* out_file = NULL;

	const char* func_name = NULL;

	struct arg_s args[] = {

		ARG_STRING(true, &func_name, "func"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(true, &out_file, "output"),
	};

	const struct opt_s opts[] = {

		{ 'L', NULL, false, OPT_SPECIAL, help_func_calc, NULL, "", "Print a list of all supported functions" },
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	num_init();

	const int N = DIMS;
	long dims[N];

	// Find function pointer before accessing memory

	function fun = NULL;

	bool function_found = false;

	for (int i = 0; NULL != calc_table[i].name; i++) {

		if (0 == strcmp(func_name, calc_table[i].name)) {

			function_found = true;

			fun = calc_table[i].func;

			break;
		}
	}

	if (!function_found)
		error("Not supported function was called!\n");


	// Execute found function

	complex float* idata = load_cfl(in_file, N, dims);
	complex float* odata = create_cfl(out_file, N, dims);

	fun(N, dims, odata, idata);

	unmap_cfl(N, dims, idata);
	unmap_cfl(N, dims, odata);

	return 0;
}



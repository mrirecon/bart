/* Copyright 2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */


#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "num/init.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/opts.h"

#ifndef DIMS
#define DIMS 16
#endif



static const char help_str[] = "Compute ROI statistics.";



int main_roistat(int argc, char* argv[argc])
{
	const char* roi_file = NULL;
	const char* in_file = NULL;
	const char* out_file = NULL;

	struct arg_s args[] = {

		ARG_INFILE(true, &roi_file, "roi"),
		ARG_INFILE(true, &in_file, "input"),
		ARG_OUTFILE(false, &out_file, "output"),
	};

	bool bessel = false;

	enum stat { ALL, COUNT, SUM, MEAN, STD, VAR, ENERGY } stat = ALL;

	const struct opt_s opts[] = {

		OPT_SET('b', &bessel, "Bessel's correction, i.e. 1 / (n - 1)"),
		OPT_SELECT('C', enum stat, &stat, COUNT, "voxel count"),
		OPT_SELECT('S', enum stat, &stat, SUM, "sum"),
		OPT_SELECT('M', enum stat, &stat, MEAN, "mean"),
		OPT_SELECT('D', enum stat, &stat, STD, "standard deviation"),
		OPT_SELECT('E', enum stat, &stat, ENERGY, "energy"),
		OPT_SELECT('V', enum stat, &stat, VAR, "variance"),
	};

	cmdline(&argc, argv, ARRAY_SIZE(args), args, help_str, ARRAY_SIZE(opts), opts);

	if (bessel && !((STD == stat) || (VAR == stat)))
		error("Bessel's correction makes sense only for variance or standard deviation");


	num_init();

	long rdims[DIMS];
	long idims[DIMS];

	complex float* roi = load_cfl(roi_file, DIMS, rdims);
	complex float* in = load_cfl(in_file, DIMS, idims);


	const char* pat_name = NULL;
	const char* avg_name = NULL;
	const char* var_name = NULL;

	if (NULL != out_file) {

		switch (stat) {

		case COUNT:
			pat_name = out_file;
			break;

		case SUM:
		case MEAN:
			avg_name = out_file;
			break;

		case STD:
		case VAR:
		case ENERGY:
			var_name = out_file;
			break;

		case ALL:

			error("No output file allowed.\n");
			break;
		}
	}


	if (!md_check_compat(DIMS, ~0UL, rdims, idims))
		error("Incompatible dimensions\n");


	long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

	long rstrs[DIMS];
	md_calc_strides(DIMS, rstrs, rdims, CFL_SIZE);


	unsigned long rflags = md_nontriv_dims(DIMS, rdims);
	unsigned long iflags = md_nontriv_dims(DIMS, idims);

	long mdims[DIMS];
	md_merge_dims(DIMS, mdims, idims, rdims);

	long odims[DIMS];
	md_select_dims(DIMS, rflags ^ iflags, odims, mdims);

	long ostrs[DIMS];
	md_calc_strides(DIMS, ostrs, odims, CFL_SIZE);


	debug_print_dims(DP_DEBUG1, DIMS, odims);


	long sdims[DIMS];
	md_singleton_dims(DIMS, sdims);

	long sstrs[DIMS];
	md_singleton_strides(DIMS, sstrs);

	complex float* pat = (pat_name ? create_cfl : anon_cfl)(pat_name, DIMS, odims);

	md_clear(DIMS, odims, pat, CFL_SIZE);
	md_zaxpy2(DIMS, mdims, ostrs, pat, 1., rstrs, roi);

	if (COUNT == stat)
		goto out1;


	complex float* avg = (avg_name ? create_cfl : anon_cfl)(avg_name, DIMS, odims);

	md_clear(DIMS, odims, avg, CFL_SIZE);
	md_zfmac2(DIMS, mdims, ostrs, avg, rstrs, roi, istrs, in);

	if (SUM == stat)
		goto out2;

	md_zdiv(DIMS, odims, avg, avg, pat);

	if (MEAN == stat)
		goto out2;

	if (bessel)
		md_zsub2(DIMS, odims, ostrs, pat, ostrs, pat, sstrs, (complex float[1]){ 1. });


	complex float* var = (var_name ? create_cfl : anon_cfl)(var_name, DIMS, odims);


	long ridims[DIMS];
	md_select_dims(DIMS, iflags | rflags, ridims, mdims);

	long ristrs[DIMS];
	md_calc_strides(DIMS, ristrs, ridims, CFL_SIZE);

	complex float* tmp = md_calloc(DIMS, ridims, CFL_SIZE);

	{
		md_zsub2(DIMS, mdims, ristrs, tmp, istrs, in, ostrs, avg);
		md_zmul2(DIMS, mdims, ristrs, tmp, ristrs, tmp, rstrs, roi);

		md_clear(DIMS, odims, var, CFL_SIZE);
		md_zfmacc2(DIMS, mdims, ostrs, var, ristrs, tmp, ristrs, tmp);
	}

	md_free(tmp);

	if (ENERGY == stat)
		goto out3;

	md_zdiv(DIMS, odims, var, var, pat);

	if (VAR == stat)
		goto out3;

	md_zsqrt(DIMS, odims, var, var);

	if (STD == stat)
		goto out3;

	assert(ALL == stat);

	long pos[DIMS] = { 0 };


	do {
		print_dims(DIMS, pos);

		bart_printf("#%12s\t%6s\t%6s\n", "mean", "std", "count");

		do {
			long count = crealf(MD_ACCESS(DIMS, ostrs, pos, pat)) + (bessel ? 1 : 0);
			complex float mn = MD_ACCESS(DIMS, ostrs, pos, avg);
			float std = crealf(MD_ACCESS(DIMS, ostrs, pos, var));

			bart_printf("%+6.3f%+.3fi\t%.3f\t%6ld\n", crealf(mn), cimagf(mn), std, count);

		} while (md_next(DIMS, odims, iflags, pos));

	} while (md_next(DIMS, odims, rflags, pos));

out3:
	unmap_cfl(DIMS, odims, var);
out2:
	unmap_cfl(DIMS, odims, avg);
out1:
	unmap_cfl(DIMS, odims, pat);
	unmap_cfl(DIMS, rdims, roi);
	unmap_cfl(DIMS, idims, in);

	return 0;
}





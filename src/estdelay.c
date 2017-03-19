
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/subpixel.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"

#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "dim <arg1> <arg2>";
static const char help_str[] = "Estimate sub-pixel shift.";

// float fit_quadratic_form(int N, float qf[3], 

int main_estdelay(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 2, usage_str, help_str);

	long dims[DIMS];
	const complex float* in = load_cfl(argv[2], DIMS, dims);
 
	unsigned int d = atoi(argv[1]);
	unsigned int flags = (1 << d);

	printf("%d\n", d);

	long dims1[DIMS];
	md_select_dims(DIMS, ~flags, dims1, dims);
	print_dims(DIMS, dims1);

	complex float* tmp1 = md_alloc(DIMS, dims1, CFL_SIZE);
	complex float* tmp2 = md_alloc(DIMS, dims1, CFL_SIZE);

	long pos[DIMS] = { 0 };

	for (unsigned int i = 0; i < dims[d]; i++) {

		pos[d] = i;
		md_copy_block(DIMS, pos, dims1, tmp1, dims, in, CFL_SIZE);

		for (unsigned int j = 0; j < dims[d]; j++) {

			pos[d] = j;
			md_copy_block(DIMS, pos, dims1, tmp2, dims, in, CFL_SIZE);

			unsigned int d2 = 0;
			float shifts[DIMS];
			est_subpixel_shift(DIMS, shifts, dims1, 1, tmp1, tmp2);

			printf("\t%f", shifts[d2]);
		}

		printf("\n");
	}

	printf("\n");
	md_free(tmp1);
	md_free(tmp2);

	unmap_cfl(DIMS, dims, in);

	exit(0);
}



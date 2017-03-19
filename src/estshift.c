
#include <complex.h>
#include <stdlib.h>
#include <stdio.h>

#include "num/multind.h"

#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/subpixel.h"


#ifndef DIMS
#define DIMS 16
#endif


static const char usage_str[] = "flags <arg1> <arg2>";
static const char help_str[] = "Estimate sub-pixel shift.";




int main_estshift(int argc, char* argv[])
{
	mini_cmdline(argc, argv, 3, usage_str, help_str);

	unsigned int flags = atoi(argv[1]);

	long dims1[DIMS];
	long dims2[DIMS];

	const complex float* in1 = load_cfl(argv[2], DIMS, dims1);
	const complex float* in2 = load_cfl(argv[3], DIMS, dims2);

	assert(md_check_compat(DIMS, ~0u, dims1, dims2));

	float shifts[DIMS];
	est_subpixel_shift(DIMS, shifts, dims1, flags, in1, in2);

	printf("Shifts:");

	for (unsigned int i = 0; i < DIMS; i++) {

		if (!MD_IS_SET(flags, i))
			continue;

		printf("\t%f", shifts[i]);
	}

	printf("\n");

	unmap_cfl(DIMS, dims1, in1);
	unmap_cfl(DIMS, dims2, in2);

	exit(0);
}



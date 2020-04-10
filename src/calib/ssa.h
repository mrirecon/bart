
#include <complex.h>

#include "misc/mri.h"

extern void ssa_fary(	const long kernel_dims[3],
			const long cal_dims[DIMS],
			const complex float* cal,
			const char* name_EOF,
			const char* name_S,
			const char* backproj,
			const int rank,
			const long group);


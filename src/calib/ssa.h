
#include <complex.h>

#include "misc/mri.h"


extern void ssa_fary(	const long kernel_dims[3],
			const long cal_dims[DIMS],
			const long A_dims[2],
			const complex float* A,
			complex float* U,
			float* S_square,
			complex float* back,
			const int rank,
			const long group);


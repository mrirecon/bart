
#include "num/vecops.h"
#include "num/gpuops.h"

#include "misc/misc.h"

#include "vec.h"


// defined in vecops.c and gpuops.c





const struct vec_iter_s* select_vecops(const float* x)
{
#ifdef USE_CUDA
	return cuda_ondevice(x) ? &gpu_iter_ops : &cpu_iter_ops;
#else
	UNUSED(x);
	return &cpu_iter_ops;
#endif
}


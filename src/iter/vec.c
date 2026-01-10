/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2022. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include "num/vecops.h"
#include "num/gpuops.h"
#include "num/vptr.h"
#include "num/vptr_iter.h"


#include "vec.h"


// defined in vecops.c and gpuops.c





const struct vec_iter_s* select_vecops(const float* x)
{
	if (is_vptr(x)) {

#ifdef USE_CUDA
		return cuda_ondevice(x) ? &vptr_iter_ops_gpu : &vptr_iter_ops;
#else
		return &vptr_iter_ops;
#endif
	}

#ifdef USE_CUDA
	return cuda_ondevice(x) ? &gpu_iter_ops : &cpu_iter_ops;
#else
	(void)x;
	return &cpu_iter_ops;
#endif
}


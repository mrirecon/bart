/* Copyright 2021. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */
 
#include "misc/misc.h"
#include "misc/types.h"

#include "num/multind.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "num/multiplace.h"


struct multiplace_array_s {

	int N;
	const long* dims;
	size_t size;

	void* ptr_ref;

	void* ptr_cpu;

#ifdef USE_CUDA
	void* ptr_gpu[MAX_CUDA_DEVICES];
#endif
	bool free;
};


static struct multiplace_array_s* multiplace_alloc(int D, const long dimensions[D], size_t size)
{
	PTR_ALLOC(struct multiplace_array_s, result);

	result->N = D;
	result->size = size;

	PTR_ALLOC(long[D], dims);

	md_copy_dims(D, *dims, dimensions);

	result->dims = *PTR_PASS(dims);

	result->ptr_cpu = NULL;
	result->free = true;

#ifdef USE_CUDA
	for (int i = 0; i < MAX_CUDA_DEVICES; i++)
		result->ptr_gpu[i] = NULL;
#endif

	return PTR_PASS(result);
}



void multiplace_free(const struct multiplace_array_s* ptr)
{
	if (NULL == ptr)
		return;

	if (ptr->free)
		md_free(ptr->ptr_cpu);

#ifdef USE_CUDA
	for (int i = 0; i < MAX_CUDA_DEVICES; i++)
		md_free(ptr->ptr_gpu[i]);
#endif

	xfree(ptr->dims);
	xfree(ptr);
}


const void* multiplace_read(struct multiplace_array_s* ptr, const void* ref)
{
	if (NULL == ptr)
		return NULL;

#ifdef USE_CUDA
	if (cuda_ondevice(ref)) {

		#pragma omp critical (bart_multiplace)
		if (NULL == ptr->ptr_gpu[cuda_get_device()]) {

			ptr->ptr_gpu[cuda_get_device()] = md_gpu_move(ptr->N, ptr->dims, ptr->ptr_ref, ptr->size);
			cuda_sync_device();
		}

		return ptr->ptr_gpu[cuda_get_device()];
	}
#else
	UNUSED(ref);
#endif
	if (NULL == ptr->ptr_cpu) {

		ptr->ptr_cpu = md_alloc(ptr->N, ptr->dims, ptr->size);

		md_copy(ptr->N, ptr->dims, ptr->ptr_cpu, ptr->ptr_ref, ptr->size);
	}

	return ptr->ptr_cpu;
}


struct multiplace_array_s* multiplace_move2(int D, const long dimensions[D], const long strides[D], size_t size, const void* ptr)
{
	auto result = multiplace_alloc(D, dimensions, size);

	void* tmp = md_alloc_sameplace(D, dimensions, size, ptr);

	md_copy2(D, dimensions, MD_STRIDES(D, dimensions, size), tmp, strides, ptr, size);

	result->ptr_ref = tmp;

#ifdef USE_CUDA
	if (cuda_ondevice(tmp))
		result->ptr_gpu[cuda_get_device()] = tmp;
	else
#endif
	result->ptr_cpu = tmp;

	return result;
}


struct multiplace_array_s* multiplace_move(int D, const long dimensions[D], size_t size, const void* ptr)
{
	return multiplace_move2(D, dimensions, MD_STRIDES(D, dimensions, size), size, ptr);
}

struct multiplace_array_s* multiplace_move_F(int D, const long dimensions[D], size_t size, const void* ptr)
{

	auto result = multiplace_alloc(D, dimensions, size);
	result->ptr_ref = (void*)ptr;

	#pragma omp critical (bart_multiplace)
	{
	#ifdef USE_CUDA
		if (cuda_ondevice(ptr)) {

			result->ptr_gpu[cuda_get_device_num(ptr)] = (void*)ptr;
			cuda_sync_device();
		} else 
	#endif
		result->ptr_cpu = (void*)ptr;
	}

	return result;
}

struct multiplace_array_s* multiplace_move_wrapper(int D, const long dimensions[D], size_t size, const void* ptr)
{

#ifdef USE_CUDA
	assert (!cuda_ondevice(ptr));
#endif

	auto result = multiplace_alloc(D, dimensions, size);
	result->ptr_cpu = (void*)ptr;
	result->ptr_ref = (void*)ptr;
	result->free = false;
	return result;
}



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

#include "num/vptr.h"

#include "num/multiplace.h"


struct multiplace_array_s {

	int N;
	const long* dims;
	size_t size;

	void* ptr_ref;

	void* ptr_cpu;
	void* mpi_cpu;

#ifdef USE_CUDA
	void* ptr_gpu[MAX_CUDA_DEVICES];
	void* mpi_gpu[MAX_CUDA_DEVICES];
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
	result->mpi_cpu = NULL;
	result->free = true;

#ifdef USE_CUDA
	for (int i = 0; i < MAX_CUDA_DEVICES; i++) {

		result->ptr_gpu[i] = NULL;
		result->mpi_gpu[i] = NULL;
	}
#endif

	return PTR_PASS(result);
}



void multiplace_free(const struct multiplace_array_s* _ptr)
{
	if (NULL == _ptr)
		return;

	struct multiplace_array_s* ptr = (struct multiplace_array_s*)_ptr;

	if (ptr->ptr_ref == ptr->ptr_cpu)
		ptr->ptr_cpu = NULL;
	
	if (ptr->ptr_ref == ptr->mpi_cpu)
		ptr->mpi_cpu = NULL;

#ifdef USE_CUDA
	for (int i = 0; i < MAX_CUDA_DEVICES; i++){

		if (ptr->ptr_ref == ptr->ptr_gpu[i])
			ptr->ptr_gpu[i] = NULL;
	
		if (ptr->ptr_ref == ptr->mpi_gpu[i])
			ptr->mpi_gpu[i] = NULL;
	}
#endif

	md_free(ptr->ptr_cpu);
	md_free(ptr->mpi_cpu);

#ifdef USE_CUDA
	for (int i = 0; i < MAX_CUDA_DEVICES; i++){

		md_free(ptr->ptr_gpu[i]);
		md_free(ptr->mpi_gpu[i]);
	}
#endif

	if (ptr->free)
		md_free(ptr->ptr_ref);

	xfree(ptr->dims);
	xfree(ptr);
}


const void* multiplace_read(struct multiplace_array_s* ptr, const void* ref)
{
	if (NULL == ptr)
		return NULL;

	if (is_vptr(ref)) {

		if (is_vptr_cpu(ref)) {

			if (NULL == ptr->mpi_cpu) {

				ptr->mpi_cpu = vptr_alloc_sameplace(ptr->N, ptr->dims, ptr->size, ref);
				md_copy(ptr->N, ptr->dims, ptr->mpi_cpu, ptr->ptr_ref, ptr->size);
			}

			return ptr->mpi_cpu;
		}

#ifdef USE_CUDA
		if (is_vptr_gpu(ref)) {

			if (NULL == ptr->mpi_gpu[cuda_get_device()]) {

				ptr->mpi_gpu[cuda_get_device()] = vptr_alloc_sameplace(ptr->N, ptr->dims, ptr->size, ref);
				md_copy(ptr->N, ptr->dims, ptr->mpi_gpu[cuda_get_device()], ptr->ptr_ref, ptr->size);
			}

			return ptr->mpi_gpu[cuda_get_device()];
		}
#endif
	} else {

#ifdef USE_CUDA
		if (cuda_ondevice(ref)) {
			
			if (NULL == ptr->ptr_gpu[cuda_get_device()]) {

				ptr->ptr_gpu[cuda_get_device()] = md_alloc_gpu(ptr->N, ptr->dims, ptr->size);
				md_copy(ptr->N, ptr->dims, ptr->ptr_gpu[cuda_get_device()], ptr->ptr_ref, ptr->size);
			}

			return ptr->ptr_gpu[cuda_get_device()];
		}
#endif
	
		if (NULL == ptr->ptr_cpu) {

			ptr->ptr_cpu = md_alloc(ptr->N, ptr->dims, ptr->size);
			md_copy(ptr->N, ptr->dims, ptr->mpi_cpu, ptr->ptr_ref, ptr->size);
		}

		return ptr->ptr_cpu;
	}

	return NULL;
}


struct multiplace_array_s* multiplace_move2(int D, const long dimensions[D], const long strides[D], size_t size, const void* ptr)
{
	auto result = multiplace_alloc(D, dimensions, size);

	void* tmp = md_alloc_sameplace(D, dimensions, size, ptr);

	md_copy2(D, dimensions, MD_STRIDES(D, dimensions, size), tmp, strides, ptr, size);

	result->ptr_ref = tmp;

	if (is_vptr(tmp)) {

		if (is_vptr_cpu(tmp))
			result->mpi_cpu = tmp;

#ifdef USE_CUDA
		if (is_vptr_gpu(tmp))
			result->mpi_gpu[cuda_get_device()] = tmp;
#endif
	} else {

#ifdef USE_CUDA
		if (cuda_ondevice(tmp))
			result->ptr_gpu[cuda_get_device()] = tmp;
		else
#endif
			result->ptr_cpu = tmp;
	}

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

	if (is_vptr(ptr)) {

		if (is_vptr_cpu(ptr))
			result->mpi_cpu = (void*)ptr;

#ifdef USE_CUDA
		if (is_vptr_gpu(ptr))
			result->mpi_gpu[cuda_get_device()] = (void*)ptr;
#endif
	} else {

#ifdef USE_CUDA
		if (cuda_ondevice(ptr))
			result->ptr_gpu[cuda_get_device()] = (void*)ptr;
		else
#endif
			result->ptr_cpu = (void*)ptr;
	}
	return result;
}

struct multiplace_array_s* multiplace_move_wrapper(int D, const long dimensions[D], size_t size, const void* ptr)
{
	auto result = multiplace_move_F(D, dimensions, size, ptr);
	result->free = false;
	return result;
}



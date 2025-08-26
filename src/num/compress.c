/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/types.h"

#include "num/multind.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls_copy.h"
#endif
#include "num/vptr_fun.h"

#include "compress.h"



long md_compress_mask_to_index(long N, const long dims[N], long* index, const complex float* mask)
{
	long tot = md_calc_size(N, dims);
	long max = 0;

	for (long i = 0; i < tot; i++)
		if (0. != crealf(mask[i]))
			index[i] = (max++);
		else
			index[i] = -1;

	return max;
}

void md_compress_dims(long N, long cdims[N], const long dcdims[N], const long mdims[N], long max)
{
	md_select_dims(N, ~md_nontriv_dims(N, mdims), cdims, dcdims);

	int cdim = ffs(md_nontriv_dims(N, mdims)) - 1;
	cdims[cdim] = max;
}

void md_decompress_dims(long N, long dcdims[N], const long cdims[N], const long mdims[N])
{
	int cdim = ffs(md_nontriv_dims(N, mdims)) - 1;

	md_select_dims(N, ~MD_BIT(cdim), dcdims, cdims);
	md_max_dims(N, ~0UL, dcdims, dcdims, mdims);
}

static void decompress_kern(long stride, long N, long dcstrs, void* dst, long istrs, const long* index, const void* src, size_t size)
{
	for (int i = 0; i < N; i++)
		if (index[i * istrs] >= 0)
			memcpy(dst + dcstrs * i, src + index[istrs * i] * stride, size);
}

static void compress_kern(long stride, long N, void* dst, long istrs, const long* index, long dcstrs, const void* src, size_t size)
{
	for (int i = 0; i < N; i++)
		if (index[i * istrs] >= 0)
			memcpy(dst + index[istrs * i] * stride, src + dcstrs * i, size);
}


struct vptr_decompress_s {

	vptr_fun_data_t super;
	size_t size;
};

DEF_TYPEID(vptr_decompress_s);


static void md_decompress2_int(vptr_fun_data_t* _data, int N, int D, const long* dims[N], const long* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_decompress_s, _data);

	const long* odims = dims[0];
	const long* idims = dims[1];
	const long* mdims = dims[2];

	const long* ostrs = strs[0];
	const long* istrs = strs[1];
	const long* mstrs = strs[2];

	void* dst = args[0];
	const void* src = args[1];
	const long* index = args[2];

	assert(1 >= bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, idims)));

	int flat_idx = 0;
	for (int i = 0; i < D; i++)
		if (1 != mdims[i] && 1 != idims[i])
			flat_idx = i;

	long midx = ffs(md_nontriv_dims(D, mdims)) - 1;
	if (0 == (bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, idims))))
		flat_idx = midx;

	long istrs2[D];
	md_select_strides(D, ~MD_BIT(flat_idx), istrs2, istrs);

	long pos[D];
	md_set_dims(D, pos, 0);

	unsigned long merge_flags = MD_BIT(midx);
	long merge_size = mdims[midx];

	for (int i = midx + 1; i < D; i++) {

		if (mdims[i] != odims[i])
			break;

		if (1 == mdims[i])
			continue;

		if ((mstrs[i] != mstrs[midx] * merge_size) || (ostrs[i] != ostrs[midx] * merge_size))
			break;

		merge_flags |= MD_BIT(i);
		merge_size *= mdims[i];
	}

#ifdef USE_CUDA
	bool gpu = cuda_ondevice(dst);
	assert(gpu == cuda_ondevice(src));
	assert(gpu == cuda_ondevice(index));
#endif

	do {
		void* tdst = dst + md_calc_offset(D, ostrs, pos);
		const void* tsrc = src + md_calc_offset(D, istrs2, pos);
		const long* tindex = &MD_ACCESS(D, mstrs, pos, index);

#ifdef USE_CUDA
		if (gpu)
			cuda_decompress(istrs[flat_idx], merge_size, ostrs[midx], tdst, mstrs[midx] / (long)sizeof(long), tindex, tsrc, d->size);
		else
#endif
		decompress_kern(istrs[flat_idx], merge_size, ostrs[midx], tdst, mstrs[midx] / (long)sizeof(long), tindex, tsrc, d->size);

	} while (md_next(D, odims, ~merge_flags, pos));
}

struct vptr_compress_s {

	vptr_fun_data_t super;
	size_t size;
};

DEF_TYPEID(vptr_compress_s);


static void md_compress2_int(vptr_fun_data_t* _data, int N, int D, const long* dims[N], const long* strs[N], void* args[N])
{
	auto d = CAST_DOWN(vptr_compress_s, _data);

	const long* odims = dims[0];
	const long* idims = dims[1];
	const long* mdims = dims[2];

	const long* ostrs = strs[0];
	const long* istrs = strs[1];
	const long* mstrs = strs[2];

	void* dst = args[0];
	const void* src = args[1];
	const long* index = args[2];


	assert(1 >= bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, odims)));

	int flat_idx = 0;
	for (int i = 0; i < D; i++)
		if (1 != mdims[i] && 1 != odims[i])
			flat_idx = i;

	long midx = ffs(md_nontriv_dims(D, mdims)) - 1;
	if (0 == (bitcount(md_nontriv_dims(D, mdims) & md_nontriv_dims(D, odims))))
		flat_idx = midx;

	long ostrs2[D];
	md_select_strides(D, ~MD_BIT(flat_idx), ostrs2, ostrs);

	long pos[D];
	md_set_dims(D, pos, 0);

	unsigned long merge_flags = MD_BIT(midx);
	long merge_size = mdims[midx];

	for (int i = midx + 1; i < D; i++) {

		if (mdims[i] != idims[i])
			break;

		if (1 == mdims[i])
			continue;

		if ((mstrs[i] != mstrs[midx] * merge_size) || (istrs[i] != istrs[midx] * merge_size))
			break;

		merge_flags |= MD_BIT(i);
		merge_size *= mdims[i];
	}

#ifdef USE_CUDA
	bool gpu = cuda_ondevice(dst);
	assert(gpu == cuda_ondevice(src));
	assert(gpu == cuda_ondevice(index));
#endif


	do {
		void* tdst = dst + md_calc_offset(D, ostrs2, pos);
		const void* tsrc = src + md_calc_offset(D, istrs, pos);
		const long* tindex = &MD_ACCESS(D, mstrs, pos, index);

#ifdef USE_CUDA
		if (gpu)
			cuda_compress(ostrs[flat_idx], merge_size, tdst, mstrs[midx] / (long)sizeof(long), tindex, istrs[midx], tsrc, d->size);
		else
#endif
		compress_kern(ostrs[flat_idx], merge_size, tdst, mstrs[midx] / (long)sizeof(long), tindex, istrs[midx], tsrc, d->size);

	} while (md_next(D, idims, ~merge_flags, pos));
}



void md_decompress2(int N, const long odims[N], const long ostrs[N], void* dst, const long idims[N], const long istrs[N], const void* src, const long mdims[N], const long mstrs[N], const long* index, const void* fill, size_t size)
{
	if (NULL != fill)
		md_fill2(N, odims, ostrs, dst, fill, size);

	PTR_ALLOC(struct vptr_decompress_s, _d);
	SET_TYPEID(vptr_decompress_s, _d);
	_d->super.del = NULL;
	_d->size = size;

	unsigned long lflags = ~md_nontriv_strides(N, mstrs);

	// 0 is read as not completely over written
	exec_vptr_fun_gen(md_decompress2_int, CAST_UP(PTR_PASS(_d)), 3, N, lflags, MD_BIT(0), MD_BIT(0) | MD_BIT(1) | MD_BIT(2),
				(const long*[3]) { odims, idims, mdims }, (const long*[3]) { ostrs, istrs, mstrs },
				(void* [3]) { dst, (void*)src, (void*)index}, (size_t[3]) { size, size, sizeof(long)}, true);
}


void md_compress2(int N, const long odims[N], const long ostrs[N], void* dst, const long idims[N], const long istrs[N], const void* src, const long mdims[N], const long mstrs[N], const long* index, size_t size)
{
	PTR_ALLOC(struct vptr_compress_s, _d);
	SET_TYPEID(vptr_compress_s, _d);
	_d->super.del = NULL;
	_d->size = size;

	unsigned long lflags = ~md_nontriv_strides(N, mstrs);

	exec_vptr_fun_gen(md_compress2_int, CAST_UP(PTR_PASS(_d)), 3, N, lflags, MD_BIT(0), MD_BIT(1) | MD_BIT(2),
				(const long*[3]) { odims, idims, mdims }, (const long*[3]) { ostrs, istrs, mstrs },
				(void* [3]) { dst, (void*)src, (void*)index}, (size_t[3]) { size, size, sizeof(long)}, true);
}


void md_decompress(int N, const long odims[N], void* dst, const long idims[N], const void* src, const long mdims[N], const long* index, const void* fill, size_t size)
{
	md_decompress2(N, odims, MD_STRIDES(N, odims, size), dst, idims, MD_STRIDES(N, idims, size), src, mdims, MD_STRIDES(N, mdims, sizeof(long)), index, fill, size);
}

void md_compress(int N, const long odims[N], void* dst, const long idims[N], const void* src, const long mdims[N], const long* index, size_t size)
{
	md_compress2(N, odims, MD_STRIDES(N, odims, size), dst, idims, MD_STRIDES(N, idims, size), src, mdims, MD_STRIDES(N, mdims, sizeof(long)), index, size);
}




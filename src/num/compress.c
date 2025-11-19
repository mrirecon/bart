/* Copyright 2024. TU Graz. Institute of Biomedical Imaging.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */

#include <complex.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/multind.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls_copy.h"
#endif

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


void md_decompress2(int N, const long odims[N], const long ostrs[N], void* dst, const long idims[N], const long istrs[N], const void* src, const long mdims[N], const long mstrs[N], const long* index, const void* fill, size_t size)
{
	assert(1 == bitcount(md_nontriv_dims(N, mdims) & md_nontriv_dims(N, idims)));

	int flat_idx = 0;
	for (int i = 0; i < N; i++)
		if (1 != mdims[i] && 1 != idims[i])
			flat_idx = i;

	long istrs2[N];
	md_select_strides(N, ~MD_BIT(flat_idx), istrs2, istrs);

	if (NULL != fill)
		md_fill2(N, odims, ostrs, dst, fill, size);

	long pos[N];
	md_set_dims(N, pos, 0);

	long midx = ffs(md_nontriv_dims(N, mdims)) - 1;

	unsigned long merge_flags = MD_BIT(midx);
	long merge_size = mdims[midx];

	for (int i = midx + 1; i < N; i++) {

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
		void* tdst = dst + md_calc_offset(N, ostrs, pos);
		const void* tsrc = src + md_calc_offset(N, istrs2, pos);
		const long* tindex = &MD_ACCESS(N, mstrs, pos, index);

#ifdef USE_CUDA
		if (gpu)
			cuda_decompress(istrs[flat_idx], merge_size, ostrs[midx], tdst, mstrs[midx] / (long)sizeof(long), tindex, tsrc, size);
		else
#endif
		decompress_kern(istrs[flat_idx], merge_size, ostrs[midx], tdst, mstrs[midx] / (long)sizeof(long), tindex, tsrc, size);

	} while (md_next(N, odims, ~merge_flags, pos));
}

void md_compress2(int N, const long odims[N], const long ostrs[N], void* dst, const long idims[N], const long istrs[N], const void* src, const long mdims[N], const long mstrs[N], const long* index, size_t size)
{
	assert(1 == bitcount(md_nontriv_dims(N, mdims) & md_nontriv_dims(N, odims)));

	int flat_idx = 0;
	for (int i = 0; i < N; i++)
		if (1 != mdims[i] && 1 != odims[i])
			flat_idx = i;

	long ostrs2[N];
	md_select_strides(N, ~MD_BIT(flat_idx), ostrs2, ostrs);

	long pos[N];
	md_set_dims(N, pos, 0);

	long midx = ffs(md_nontriv_dims(N, mdims)) - 1;

	unsigned long merge_flags = MD_BIT(midx);
	long merge_size = mdims[midx];

	for (int i = midx + 1; i < N; i++) {

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
		void* tdst = dst + md_calc_offset(N, ostrs2, pos);
		const void* tsrc = src + md_calc_offset(N, istrs, pos);
		const long* tindex = &MD_ACCESS(N, mstrs, pos, index);

#ifdef USE_CUDA
		if (gpu)
			cuda_compress(ostrs[flat_idx], merge_size, tdst, mstrs[midx] / (long)sizeof(long), tindex, istrs[midx], tsrc, size);
		else
#endif
		compress_kern(ostrs[flat_idx], merge_size, tdst, mstrs[midx] / (long)sizeof(long), tindex, istrs[midx], tsrc, size);

	} while (md_next(N, idims, ~merge_flags, pos));
}

void md_decompress(int N, const long odims[N], void* dst, const long idims[N], const void* src, const long mdims[N], const long* index, const void* fill, size_t size)
{
	md_decompress2(N, odims, MD_STRIDES(N, odims, size), dst, idims, MD_STRIDES(N, idims, size), src, mdims, MD_STRIDES(N, mdims, sizeof(long)), index, fill, size);
}

void md_compress(int N, const long odims[N], void* dst, const long idims[N], const void* src, const long mdims[N], const long* index, size_t size)
{
	md_compress2(N, odims, MD_STRIDES(N, odims, size), dst, idims, MD_STRIDES(N, idims, size), src, mdims, MD_STRIDES(N, mdims, sizeof(long)), index, size);
}




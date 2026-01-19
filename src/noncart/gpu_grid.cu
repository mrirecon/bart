/* Copyright 2022. Uecker Lab. University Medical Center Göttingen.
 * Copyright 2022-2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include <assert.h>
#include <stdbool.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/gpukrnls_misc.h"
#include "num/gpuops.h"
#include "num/multiplace.h"

#include "noncart/grid.h"
#include "gpu_grid.h"


struct linphase_conf {

	long dims[3];
	long tot;
	float shifts[3];
	long N;
	float cn;
	float scale;
	_Bool conj;
	_Bool fmac;
};

template <_Bool fmac>
__global__ void kern_apply_linphases_3D(struct linphase_conf c, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int startX = threadIdx.x + blockDim.x * blockIdx.x;
	int strideX = blockDim.x * gridDim.x;

	int startY = threadIdx.y + blockDim.y * blockIdx.y;
	int strideY = blockDim.y * gridDim.y;

	int startZ = threadIdx.z + blockDim.z * blockIdx.z;
	int strideZ = blockDim.z * gridDim.z;

	for (long z = startZ; z < c.dims[2]; z += strideZ)
		for (long y = startY; y < c.dims[1]; y += strideY)
			for (long x = startX; x < c.dims[0]; x +=strideX) {

				long pos[3] = { x, y, z };
				long idx = x + c.dims[0] * (y + c.dims[1] * z);

				float val = c.cn;

				for (int n = 0; n < 3; n++)
					val += pos[n] * c.shifts[n];

				if (c.conj)
					val = -val;

				float si;
				float co;
				sincosf(val, &si, &co);

				cuFloatComplex cval = make_cuFloatComplex(c.scale * co, c.scale * si);

				if (fmac) {

					for (long i = 0; i < c.N; i++)
						dst[idx + i * c.tot] = cuCaddf(dst[idx + i * c.tot], cuCmulf(src[idx + i * c.tot], cval));
				} else {

					for (long i = 0; i < c.N; i++)
						dst[idx + i * c.tot] = cuCmulf(src[idx + i * c.tot], cval);
				}
			}
}



extern "C" void cuda_apply_linphases_3D(int N, const long img_dims[], const float _shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, _Bool fftm, float scale)
{
	struct linphase_conf c;

	c.cn = 0;
	c.tot = 1;
	c.N = 1;
	c.scale = scale;
	c.conj = conj;
	c.fmac = fmac;

	float shifts[3];

	for (int i = 0; i < 3; i++) {

		shifts[i] = _shifts[i];

		if (1 < img_dims[i])
			shifts[i] += (img_dims[i] / 2. - img_dims[i] / 2);
	}

	for (int n = 0; n < 3; n++) {

		c.shifts[n] = 2. * M_PI * (float)(shifts[n]) / ((float)img_dims[n]);
		c.cn -= c.shifts[n] * (float)img_dims[n] / 2.;

		c.dims[n] = img_dims[n];
		c.tot *= c.dims[n];

		if (fftm) {

			long center = c.dims[n] / 2;
			double shift = (double)center / (double)c.dims[n];

			c.shifts[n] += 2. * M_PI * shift;
			c.cn -= 2. * M_PI * shift * (double)center / 2.;
		}
	}

	c.N = md_calc_size(N - 3, img_dims + 3);

	if (c.fmac) {

		const void* func = (const void*)kern_apply_linphases_3D<true>;
		kern_apply_linphases_3D<true><<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
	} else {

		const void* func = (const void*)kern_apply_linphases_3D<false>;
		kern_apply_linphases_3D<false><<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
	}

	CUDA_KERNEL_ERROR;
}


static __device__ double ftkb(double beta, double x)
{
	double a = pow(beta, 2.) - pow(M_PI * x, 2.);

	if (0. == a)
		return 1;

	if (a > 0)
		return (sinh(sqrt(a)) / sqrt(a));
	else
		return (sin(sqrt(-a)) / sqrt(-a));
}

static __device__ double rolloff(double x, double beta, double width)
{
	return 1. / ftkb(beta, x * width) / width;
}

static __device__ float posf(int d, int i, float os)
{
	if (1 == d)
		return 0.;

	int od = os * d;
	int oi = i + (od / 2 - d / 2);

	return (((float)oi - (float)(od / 2)) / (float)od);
}

struct rolloff_conf {

	long dims[4];
	long ostrs[4];
	long istrs[4];
	float os;
	float width;
	float beta;
	double bessel_beta;
};

__global__ void kern_apply_rolloff_correction(struct rolloff_conf c, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int startX = threadIdx.x + blockDim.x * blockIdx.x;
	int strideX = blockDim.x * gridDim.x;

	int startY = threadIdx.y + blockDim.y * blockIdx.y;
	int strideY = blockDim.y * gridDim.y;

	int startZ = threadIdx.z + blockDim.z * blockIdx.z;
	int strideZ = blockDim.z * gridDim.z;

	for (long z = startZ; z < c.dims[2]; z += strideZ)
		for (long y = startY; y < c.dims[1]; y += strideY)
			for (long x = startX; x < c.dims[0]; x +=strideX) {

				long iidx = x * c.istrs[0] + y * c.istrs[1] + z * c.istrs[2];
				long oidx = x * c.ostrs[0] + y * c.ostrs[1] + z * c.ostrs[2];

				float val = ((c.dims[0] > 1) ? rolloff(posf(c.dims[0], x, c.os), c.beta, c.width) * c.bessel_beta : 1)
					  * ((c.dims[1] > 1) ? rolloff(posf(c.dims[1], y, c.os), c.beta, c.width) * c.bessel_beta : 1)
					  * ((c.dims[2] > 1) ? rolloff(posf(c.dims[2], z, c.os), c.beta, c.width) * c.bessel_beta : 1);

				for (long i = 0; i < c.dims[3]; i++) {

					dst[oidx + i * c.ostrs[3]].x = val * src[iidx + i * c.istrs[3]].x;
					dst[oidx + i * c.ostrs[3]].y = val * src[iidx + i * c.istrs[3]].y;
				}
			}
}



extern "C" void cuda_apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[4], const long ostrs[4], _Complex float* dst, const long istrs[4], const _Complex float* src)
{
	struct rolloff_conf c;

	c.os = os,
	c.width = width,
	c.beta = beta,
	c.bessel_beta = bessel_kb_beta,

	md_copy_dims(4, c.dims, dims);
	md_copy_dims(4, c.ostrs, ostrs);
	md_copy_dims(4, c.istrs, istrs);

	const void* func = (const void*)kern_apply_rolloff_correction;
	kern_apply_rolloff_correction<<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);

	CUDA_KERNEL_ERROR;
}


// Linear interpolation
__device__ static __inline__ float lerp(float a, float b, float c)
{
	return (1. - c) * a + c * b;
}

// Linear interpolation look up
__device__ static float intlookup(int n, const float* table, float x)
{
	x *= 2;
	int index = (int)(x * (n - 1));
	float fpart = x * (n - 1) - (float)index;
	float l = lerp(table[index], table[index + 1], fpart);
	return l;
}


static const struct multiplace_array_s* kb_table = NULL;

static void kb_precompute_gpu(double beta)
{
	#pragma omp critical(kb_tbale_gpu)
	if (NULL == kb_table)
		kb_table = kb_get_table(beta);
}

struct access_bin_s {

	long Nt;
	long* sample_idx;

	// sorting into subgrids
	int shared_size;
	int local_size[3];
	int width[3];

	long grid_count;
	long* bin_offset;
	int* bin_prop;
};

struct access_stride_s {

	long ksp_dims[2];
	long ksp_strs[2];
	long trj_strs[2];
};

struct grid_plan_s {

	struct grid_conf_s conf;

	int kb_size;
	float* kb_table;

	int grd_dims[3];

	int Nc;
	long col_str_ksp;
	long col_str_grd;

	bool sort;

	union {

		struct access_bin_s bin;
		struct access_stride_s stride;
	} access;
};

void debug_grid_plan(int dl, struct grid_plan_s* plan)
{
	debug_printf(dl, "grid_plan:\n");
	debug_printf(dl, "  conf:\n");
	debug_printf(dl, "    os: %f\n", plan->conf.os);
	debug_printf(dl, "    width: %f\n", plan->conf.width);
	debug_printf(dl, "    beta: %f\n", plan->conf.beta);
	debug_printf(dl, "    shift: %f %f %f\n", plan->conf.shift[0], plan->conf.shift[1], plan->conf.shift[2]);
	debug_printf(dl, "    periodic: %d\n", plan->conf.periodic);
	debug_printf(dl, "  grid_dims: %d %d %d\n", plan->grd_dims[0], plan->grd_dims[1], plan->grd_dims[2]);
	debug_printf(dl, "  Nc: %d, ksp_strs: %ld grd_strs: %ld\n", plan->Nc, plan->col_str_ksp, plan->col_str_grd);

	if (plan->sort) {

		debug_printf(dl, "  bin:\n");
		debug_printf(dl, "    Nt: %ld\n", plan->access.bin.Nt);
		debug_printf(dl, "    grid_count: %ld\n", plan->access.bin.grid_count);
		debug_printf(dl, "    shared_size: %d\n", plan->access.bin.shared_size);
		debug_printf(dl, "    local_size: %d %d %d\n", plan->access.bin.local_size[0], plan->access.bin.local_size[1], plan->access.bin.local_size[2]);
		debug_printf(dl, "    width: %d %d %d\n", plan->access.bin.width[0], plan->access.bin.width[1], plan->access.bin.width[2]);

	} else {

		debug_printf(dl, "  stride:\n");
		debug_printf(dl, "    ksp_dims: %ld %ld\n", plan->access.stride.ksp_dims[0], plan->access.stride.ksp_dims[1]);
		debug_printf(dl, "    ksp_strs: %ld %ld\n", plan->access.stride.ksp_strs[0], plan->access.stride.ksp_strs[1]);
		debug_printf(dl, "    trj_strs: %ld %ld\n", plan->access.stride.trj_strs[0], plan->access.stride.trj_strs[1]);
	}
}

void grid_plan_free(struct grid_plan_s plan)
{
	if (plan.sort) {

		md_free(plan.access.bin.sample_idx);
		md_free(plan.access.bin.bin_offset);
		md_free(plan.access.bin.bin_prop);
	}
}

struct grid_sort_plan_s {

	struct grid_conf_s conf;
	int grd_dims[3];
	int bin_size[3];
	int bin_dims[3];

	struct access_stride_s stride;
};

__device__ static int get_bin_index(struct grid_sort_plan_s gd, float trj[3])
{
	int bin = 0;

	for (int j = 2; j >= 0; j--) {

		trj[j] = (trj[j] + gd.conf.shift[j]) * gd.conf.os;
		trj[j] += (gd.grd_dims[j] > 1) ? (float)(gd.grd_dims[j] / 2) : 0.;

		if (gd.conf.periodic) {

			while (trj[j] < 0.)
				trj[j] += gd.grd_dims[j];

			while (trj[j] >= gd.grd_dims[j])
				trj[j] -= gd.grd_dims[j];
		}

		float pos = floor(trj[j]);

		pos = MAX(pos, 0.);
		pos = MIN(floor(pos / gd.bin_size[j]), gd.bin_dims[j] - 1);

		bin *= gd.bin_dims[j];
		bin += pos;
	}

	return bin;
}


__global__ static void kern_grid_sort(struct grid_sort_plan_s gd, long2* bin_idx, unsigned long long* bin_count, const cuFloatComplex* traj)
{
	int startx = threadIdx.x + blockDim.x * blockIdx.x;
	int stridex = blockDim.x * gridDim.x;

	int starty = threadIdx.y + blockDim.y * blockIdx.y;
	int stridey = blockDim.y * gridDim.y;

	for (long y = starty; y < gd.stride.ksp_dims[1]; y += stridey) {
		for (long x = startx; x < gd.stride.ksp_dims[0]; x += stridex) {

			long toffset = x * gd.stride.trj_strs[0] + y * gd.stride.trj_strs[1];

			float trj[3] = { traj[0 + toffset].x, traj[1 + toffset].x, traj[2 + toffset].x };

			long bin = get_bin_index(gd, trj);

			long idx = x + y * gd.stride.ksp_dims[0];
			long idx_in_bin = atomicAdd(&bin_count[bin], 1);

			bin_idx[idx].x = bin;
			bin_idx[idx].y = idx_in_bin;
		}
	}
}

__global__ static void kern_grid_sort_cont(struct grid_sort_plan_s gd, long N, long2* bin_idx, unsigned long long* bin_count, const cuFloatComplex* traj)
{
	int startx = threadIdx.x + blockDim.x * blockIdx.x;
	int stridex = blockDim.x * gridDim.x;

	for (long i = startx; i < N; i += stridex) {

		float trj[3] = { traj[0 + 3 * i].x, traj[1 + 3 * i].x, traj[2 + 3 * i].x };

		long bin = get_bin_index(gd, trj);

		long idx_in_bin = atomicAdd(&bin_count[bin], 1);

		bin_idx[i].x = bin;
		bin_idx[i].y = idx_in_bin;
	}
}

static void cuda_grid_sort(struct grid_sort_plan_s gd, long2* bin_idx, unsigned long long* bin_count, const cuFloatComplex* traj)
{
	if (3 == gd.stride.trj_strs[0] && 3 * gd.stride.ksp_dims[0] == gd.stride.trj_strs[1]) {

		long N = gd.stride.ksp_dims[0] * gd.stride.ksp_dims[1];

		kern_grid_sort_cont<<<getGridSize(N, 1024), getBlockSize(N, 1024), 0, cuda_get_stream()>>>(gd, N, bin_idx, bin_count, traj);

	} else {
		const long size[3] = { gd.stride.ksp_dims[0], gd.stride.ksp_dims[1], 1 };

		dim3 cu_block = getBlockSize3(size, (const void*)kern_grid_sort);
		dim3 cu_grid = getGridSize3(size, (const void*)kern_grid_sort);

		kern_grid_sort<<<cu_grid, cu_block, 0, cuda_get_stream()>>>(gd, bin_idx, bin_count, (const cuFloatComplex*)traj);
	}

	CUDA_KERNEL_ERROR;
}



__global__ static void kern_grid_sort_invert(struct grid_sort_plan_s gd, long2* sample_idx, long2* bin_idx, unsigned long long* bin_offset)
{
	int startx = threadIdx.x + blockDim.x * blockIdx.x;
	int stridex = blockDim.x * gridDim.x;

	int starty = threadIdx.y + blockDim.y * blockIdx.y;
	int stridey = blockDim.y * gridDim.y;

	for (long y = starty; y < gd.stride.ksp_dims[1]; y += stridey) {
		for (long x = startx; x < gd.stride.ksp_dims[0]; x += stridex) {

			long sidx = x + y * gd.stride.ksp_dims[0];

			long bin = bin_idx[sidx].x;
			long idx = bin_offset[bin] + bin_idx[sidx].y;

			sample_idx[idx].x = x * gd.stride.trj_strs[0] + y * gd.stride.trj_strs[1];
			sample_idx[idx].y = x * gd.stride.ksp_strs[0] + y * gd.stride.ksp_strs[1];
		}
	}
}


static void cuda_grid_sort_invert(struct grid_sort_plan_s gd, long2* sample_idx, long2* bin_idx, unsigned long long* bin_offset)
{
	const long size[3] = { gd.stride.ksp_dims[0], gd.stride.ksp_dims[1], 1 };

	dim3 cu_block = getBlockSize3(size, (const void*)kern_grid_sort_invert);
	dim3 cu_grid = getGridSize3(size, (const void*)kern_grid_sort_invert);

	kern_grid_sort_invert<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(gd, sample_idx, bin_idx, bin_offset);

	CUDA_KERNEL_ERROR;
}



__global__ static void kern_exclusive_scan(long pre_zeros, long dim_reduce, long dim_batch, unsigned long long* sum, unsigned long long* dat)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	for (long block = bid; block < dim_batch; block += gridDim.x) {

		unsigned long long* ldat = dat + block * (dim_reduce - pre_zeros);
		ldat -= pre_zeros;

		int s = dim_reduce / 2;
		int l = 1;

		for (; s > 0; s /= 2, l *= 2) {

			for (int i = tid; i < s; i += blockDim.x) {

				int idxl = 1 * l - 1 + 2 * l * i;
				int idxu = 2 * l - 1 + 2 * l * i;

				if (idxl < pre_zeros)
					continue;

				ldat[idxu] += ldat[idxl];
			}

			__syncthreads();
		}

		if (0 == tid) {

			if (NULL != sum)
				sum[block] = ldat[l - 1];

			ldat[l - 1] = 0;
		}

		__syncthreads();

		s = 1;

		for (; s / 2 < dim_reduce; s *= 2, l /= 2) {

			for (long i = tid; i < s / 2; i += blockDim.x) {

				int idxl = 1 * l - 1 + 2 * l * i;
				int idxu = 2 * l - 1 + 2 * l * i;

				if (idxl < pre_zeros)
					continue;

				unsigned long long tmp = ldat[idxu];
				ldat[idxu] += ldat[idxl];
				ldat[idxl] = tmp;
			}

			__syncthreads();
		}
	}
}

void cuda_exclusive_scan(long dim_reduce, long dim_batch, unsigned long long* sum, unsigned long long* dat)
{
	long ldim = 1;
	while (ldim < dim_reduce)
		ldim *= 2;

	kern_exclusive_scan<<<MIN(1024, dim_batch), MIN(ldim, 1024), 0, cuda_get_stream()>>>(ldim - dim_reduce, ldim, dim_batch, sum, dat);

	CUDA_KERNEL_ERROR;
}


#define CACHE_SIZE 48 * 1024

static void grid_plan_compute_binning(int width[3], int bin_size[3], int bin_dims[3], struct grid_conf_s conf, const long grid_dims[3])
{
	for (int i = 0; i < 3; i++) {

		width[i] = ((int)ceilf(0.5 * conf.width)) * (1 == grid_dims[i] ? 0 : 1);
		bin_size[i] = 1;
		bin_dims[i] = 1;
	}

	int i = 0;
	unsigned long ext_flag = md_nontriv_dims(3, grid_dims);

	while (0 != ext_flag && (bin_size[0] < grid_dims[0] || bin_size[1] < grid_dims[1] || bin_size[2] < grid_dims[2])) {

		if (!MD_IS_SET(ext_flag, i % 3)) {

			i++;
			continue;
		}

		int tdims[3];

		for (int j = 0; j < 3; j++)
			tdims[j] = 2 * width[j] + bin_size[j];

		tdims[i % 3] += bin_size[i % 3];

		long size = tdims[0] * tdims[1] * tdims[2];

		if (size * sizeof(_Complex float) < CACHE_SIZE - ((1 + kb_size) * sizeof(float))) {

			bin_size[i % 3] *= 2;

			if (2 * bin_size[i % 3] > grid_dims[i % 3])
				ext_flag = MD_CLEAR(ext_flag, i % 3);
		} else {

			ext_flag = MD_CLEAR(ext_flag, i % 3);
		}

		i++;
	}

	for (int i = 0; i < 3; i++)
		bin_size[i] = MIN(bin_size[i], 16);

	for (int i = 0; i < 3; i++)
		bin_dims[i] = (grid_dims[i] + bin_size[i] - 1) / bin_size[i];
}

struct grid_plan_s grid_plan_create(struct grid_conf_s conf, bool sort, const long grid_dims[4], const long grid_strs[4], const long ksp_dims[4], const long ksp_strs[4], const long trj_strs[4], const _Complex float* traj)
{

	struct grid_plan_s ret = {

		.conf = conf,

		.kb_size = kb_size,
		.kb_table = (float*)multiplace_read((struct multiplace_array_s*)kb_table, (const void*)traj),

		.grd_dims = { (int)grid_dims[0], (int)grid_dims[1], (int)grid_dims[2] },
		.Nc = (int)grid_dims[3],
		.col_str_ksp = ksp_strs[3] / (long)CFL_SIZE,
		.col_str_grd = grid_strs[3] / (long)CFL_SIZE,
	};

	struct access_stride_s stride = {

		.ksp_dims = { ksp_dims[1], ksp_dims[2] },
		.ksp_strs = { ksp_strs[1] / (long)CFL_SIZE, ksp_strs[2] / (long)CFL_SIZE },
		.trj_strs = { trj_strs[1] / (long)CFL_SIZE, trj_strs[2] / (long)CFL_SIZE },
	};

	if (!sort) {

		ret.sort = false;
		ret.access.stride = stride;
		return ret;
	}

	ret.sort = true;

	struct grid_sort_plan_s sort_plan {

		.conf = conf,
		.grd_dims = { (int)grid_dims[0], (int)grid_dims[1], (int)grid_dims[2] },
		.bin_size = { 0, 0, 0 },
		.bin_dims = { 0, 0, 0 },
		.stride = stride,
	};

	int width[3];

	grid_plan_compute_binning(width, sort_plan.bin_size, sort_plan.bin_dims, conf, grid_dims);

	long bd_tot = sort_plan.bin_dims[0] * sort_plan.bin_dims[1] * sort_plan.bin_dims[2];
	size_t bin_count_size = sizeof(unsigned long long) * bd_tot;

	unsigned long long* bin_count = (unsigned long long*)cuda_malloc(bin_count_size);
	cuda_clear(bin_count_size, bin_count);

	long2* sortidx = (long2*)cuda_malloc(ksp_dims[1] * ksp_dims[2] * sizeof(long2));
	cuda_grid_sort(sort_plan, sortidx, bin_count, (const cuFloatComplex*)traj);

	unsigned long long* bin_offset = (unsigned long long*)cuda_malloc(bin_count_size);
	cuda_memcpy(bin_count_size, bin_offset, bin_count);
	cuda_exclusive_scan(bd_tot, 1, NULL, bin_offset);

	unsigned long long* bin_count_host = (unsigned long long*)xmalloc(bin_count_size);
	cuda_memcpy(bin_count_size, bin_count_host, bin_count);
	cuda_free(bin_count);

	long num_blocks = 0;
	long max_binsize = 1024 * 4;

	long max_num_blocks = sort_plan.bin_dims[0] * sort_plan.bin_dims[1] * sort_plan.bin_dims[2];
	max_num_blocks += (ksp_dims[1] * ksp_dims[2]) / max_binsize + 1;

	long* offset = (long*)xmalloc(max_num_blocks * sizeof(long));
	int* bin_prop = (int*)xmalloc(4 * max_num_blocks * sizeof(int));

	for (int z = 0; z < sort_plan.bin_dims[2]; z++)
	for (int y = 0; y < sort_plan.bin_dims[1]; y++)
	for (int x = 0; x < sort_plan.bin_dims[0]; x++) {

		long bidx = x + y * sort_plan.bin_dims[0] + z * sort_plan.bin_dims[0] * sort_plan.bin_dims[1];

		while (0 < bin_count_host[bidx]) {

			if (0 == num_blocks)
				offset[num_blocks] = 0;
			else
				offset[num_blocks] = offset[num_blocks - 1] + bin_prop[4 * (num_blocks - 1) + 0];

			bin_prop[4 * num_blocks + 0] = MIN(bin_count_host[bidx], max_binsize);
			bin_prop[4 * num_blocks + 1] = x * sort_plan.bin_size[0];
			bin_prop[4 * num_blocks + 2] = y * sort_plan.bin_size[1];
			bin_prop[4 * num_blocks + 3] = z * sort_plan.bin_size[2];

			bin_count_host[bidx] -= bin_prop[4 * num_blocks + 0];
			num_blocks++;
		}
	}

	xfree(bin_count_host);

	struct access_bin_s bin = {

		.Nt = ksp_dims[1] * ksp_dims[2],
		.sample_idx = (long*)cuda_malloc(2 * ksp_dims[1] * ksp_dims[2] * sizeof(long)),

		// sorting into subgrids
		.shared_size = 1,
		.local_size = { 1, 1, 1 },
		.width = { 0, 0, 0 },

		.grid_count = num_blocks,
		.bin_offset = (long*)cuda_malloc(num_blocks * sizeof(long)),
		.bin_prop = (int*)cuda_malloc(4 * num_blocks * sizeof(int)),
	};

	for (int i = 0; i < 3; i++) {

		bin.local_size[i] = sort_plan.bin_size[i] + 2 * width[i];
		bin.width[i] = width[i];
		bin.shared_size *= bin.local_size[i];
	}

	cuda_memcpy(num_blocks * sizeof(long), bin.bin_offset, offset);
	xfree(offset);

	cuda_memcpy(4 * num_blocks * sizeof(int), bin.bin_prop, bin_prop);
	xfree(bin_prop);

	cuda_grid_sort_invert(sort_plan, (long2*)bin.sample_idx, sortidx, bin_offset);

	cuda_free(sortidx);
	cuda_free(bin_offset);

	ret.access.bin = bin;

	return ret;
}


__device__ static inline long local_to_global_idx(const struct grid_plan_s* plan, int idx, int pos[3])
{

	pos[0] += idx % plan->access.bin.local_size[0];
	idx /= plan->access.bin.local_size[0];

	pos[1] += idx % plan->access.bin.local_size[1];
	pos[2] += idx / plan->access.bin.local_size[1];

	pos[0] -= plan->access.bin.width[0];
	pos[1] -= plan->access.bin.width[1];
	pos[2] -= plan->access.bin.width[2];

	if (plan->conf.periodic) {

		while (pos[0] < 0)
			pos[0] += plan->grd_dims[0];

		while (pos[1] < 0)
			pos[1] += plan->grd_dims[1];

		while (pos[2] < 0)
			pos[2] += plan->grd_dims[2];

		while (pos[0] >= plan->grd_dims[0])
			pos[0] -= plan->grd_dims[0];

		while (pos[1] >= plan->grd_dims[1])
			pos[1] -= plan->grd_dims[1];

		while (pos[2] >= plan->grd_dims[2])
			pos[2] -= plan->grd_dims[2];
	}

	if (0 > pos[0] || pos[0] >= plan->grd_dims[0] || 0 > pos[1] || pos[1] >= plan->grd_dims[1] || 0 > pos[2] || pos[2] >= plan->grd_dims[2])
		return -1;

	return pos[0] + pos[1] * plan->grd_dims[0] + pos[2] * plan->grd_dims[0] * plan->grd_dims[1];
}

struct grid_data_device {

	float pos[3];
	int sti[3];
	int eni[3];
	int off[3];
};

__device__ static __inline__ void dev_atomic_zadd_scl(cuFloatComplex* arg, cuFloatComplex val, float scl)
{
	if (0. == scl)
		return;

	if (0. != val.x)
		atomicAdd(&(arg->x), val.x * scl);

	if (0. != val.y)
		atomicAdd(&(arg->y), val.y * scl);
}

#define MAX_WIDTH 8

template<_Bool adjoint, _Bool smem>
__device__ static void grid_point_r(const struct grid_plan_s* plan, cuFloatComplex* grd, cuFloatComplex* ksp, const float traj[3])
{
	if (!adjoint && 0. == ksp[0].x && 0. == ksp[0].y)
		return;

	int off[3];
	int num[3];

	float d[3][MAX_WIDTH];
	assert(2 * ceil(0.5 * plan->conf.width) + 1 < MAX_WIDTH);

	for (int j = 0; j < 3; j++) {

		float pos = plan->conf.os * (traj[j] + plan->conf.shift[j]);
		pos += (plan->grd_dims[j] > 1) ? (float)(plan->grd_dims[j] / 2) : 0.;

		if (smem && plan->conf.periodic) {

			while (0. > pos)
				pos += plan->grd_dims[j];

			while (pos >= plan->access.bin.local_size[j])
				pos -= plan->grd_dims[j];
		}

		int sti = (int)ceil(pos - 0.5 * plan->conf.width);
		int eni = (int)floor(pos + 0.5 * plan->conf.width);

		off[j] = 0;

		if (smem) {

			sti = MAX(sti, 0);
			eni = MIN(eni, plan->access.bin.local_size[j] - 1);
		} else {

			if (!plan->conf.periodic) {

				sti = MAX(sti, 0);
				eni = MIN(eni, plan->grd_dims[j] - 1);

			} else {

				while (sti + off[j] < 0)
					off[j] += plan->grd_dims[j];
			}
		}

		if (1 == plan->grd_dims[j]) {

			assert(0. == pos); // ==0. fails nondeterministically for test_nufft_forward bbdec08cb
			sti = 0;
			eni = 0;
		}

		for (int i = 0; i + sti <= eni; i++) {

			assert(i < MAX_WIDTH);
			d[j][i] = intlookup(plan->kb_size, plan->kb_table, fabs(((float)(i + sti) - pos) / plan->conf.width));
		}

		num[j] = eni - sti + 1;
		off[j] += sti;
	}

	if (!adjoint && 0 == ksp[0].x && 0 == ksp[0].y)
		return;

	int ind[3];
	for (ind[2] = off[2]; ind[2] < off[2] + num[2]; ind[2]++)
	for (ind[1] = off[1]; ind[1] < off[1] + num[1]; ind[1]++)
	for (ind[0] = off[0]; ind[0] < off[0] + num[0]; ind[0]++) {

		long idx = 0;

		if (smem) {
			for (int i = 2; i >= 0; i--) {

				idx *= plan->access.bin.local_size[i];
				idx += ind[i];
			}
		} else {

			for (int i = 2; i >= 0; i--) {

				idx *= plan->grd_dims[i];
				idx += ind[i] % plan->grd_dims[i];
			}
		}

		float dp = 1.;
		for (int i = 0; i < 3; i++)
			dp *= d[i][ind[i] - off[i]];

		if (adjoint)
			dev_atomic_zadd_scl(ksp, grd[idx], dp);
		else
			dev_atomic_zadd_scl(grd + idx , ksp[0], dp);
	}
}


template<_Bool adjoint>
__global__ static void kern_grid(struct grid_plan_s plan, const cuFloatComplex* traj, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start[3];
	int stride[3];

	start[0] = threadIdx.x + blockDim.x * blockIdx.x;
	start[1] = threadIdx.y + blockDim.y * blockIdx.y;
	start[2] = threadIdx.z + blockDim.z * blockIdx.z;

	stride[0] = blockDim.x * gridDim.x;
	stride[1] = blockDim.y * gridDim.y;
	stride[2] = blockDim.z * gridDim.z;

	long pos[3];

	for (pos[2] = start[2]; pos[2] < plan.Nc; pos[2] += stride[2])
	for (pos[1] = start[1]; pos[1] < plan.access.stride.ksp_dims[1]; pos[1] += stride[1])
	for (pos[0] = start[0]; pos[0] < plan.access.stride.ksp_dims[0]; pos[0] += stride[0]) {

		long offset_trj = 0;
		long offset_ksp = 0;

		for (int i = 0; i < 2; i++) {

			offset_trj += plan.access.stride.trj_strs[i] * pos[i];
			offset_ksp += plan.access.stride.ksp_strs[i] * pos[i];
		}

		offset_ksp += plan.col_str_ksp * pos[2];

		//loop over coils
		long offset_grd = pos[2] * plan.col_str_grd;

		float trj[3] = { traj[0 + offset_trj].x, traj[1 + offset_trj].x, traj[2 + offset_trj].x };

		if (adjoint)
			grid_point_r<true, false>(&plan, (cuFloatComplex*)src + offset_grd, dst + offset_ksp, trj);
		else
			grid_point_r<false, false>(&plan, dst + offset_grd, (cuFloatComplex*)src + offset_ksp, trj);
	}
}

template<_Bool adjoint>
__global__ static void kern_grid_sorted(struct grid_plan_s plan, const cuFloatComplex* traj, cuFloatComplex* dst, const cuFloatComplex* src)
{
	extern __shared__ cuFloatComplex grd_local[];

	for (long block = blockIdx.x; block < plan.access.bin.grid_count; block += gridDim.x) {

		int4 bin_prop = ((int4*)plan.access.bin.bin_prop)[block];

		for (int c = 0; c < plan.Nc; c++) {

			for (int i = threadIdx.x; i < plan.access.bin.shared_size; i += blockDim.x) {

				if (adjoint) {

					int pos[3] = { bin_prop.y, bin_prop.z, bin_prop.w };
					long offset = local_to_global_idx(&plan, i, pos);

					if (-1 != offset)
						grd_local[i] = src[offset + c * plan.col_str_grd];
					else
						grd_local[i] = make_cuFloatComplex(0., 0.);
				} else {

					grd_local[i] = make_cuFloatComplex(0., 0.);
				}
			}

			__syncthreads();

			for (int i = threadIdx.x; i < bin_prop.x; i += blockDim.x) {

				long sample = plan.access.bin.bin_offset[block] + i;

				long2 sample_idx = ((long2*)plan.access.bin.sample_idx)[sample];

				long offset_trj = sample_idx.x;
				long offset_ksp = sample_idx.y + c * plan.col_str_ksp;

				float trj[3] = { traj[0 + offset_trj].x, traj[1 + offset_trj].x, traj[2 + offset_trj].x };

				trj[0] += -(bin_prop.y - plan.access.bin.width[0]) / plan.conf.os;
				trj[1] += -(bin_prop.z - plan.access.bin.width[1]) / plan.conf.os;
				trj[2] += -(bin_prop.w - plan.access.bin.width[2]) / plan.conf.os;

				if (adjoint)
					grid_point_r<true, true>(&plan, (cuFloatComplex*)grd_local, dst + offset_ksp, trj);
				else
					grid_point_r<false, true>(&plan, grd_local, (cuFloatComplex*)src + offset_ksp, trj);
			}

			__syncthreads();

			if (!adjoint) {

				for (int i = threadIdx.x; i < plan.access.bin.shared_size; i += blockDim.x) {

					if (0 != grd_local[i].x || 0 != grd_local[i].y) {

						int pos[3] = { bin_prop.y, bin_prop.z, bin_prop.w };
						long offset = local_to_global_idx(&plan, i, pos);

						if (-1 != offset)
							dev_atomic_zadd_scl(dst + offset + c * plan.col_str_grd, grd_local[i], 1.);
					}
				}
			}

			__syncthreads();

		}
	}
}


void cuda_grid(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long grid_dims[4], const long grid_strs[4], _Complex float* grid, const long ksp_strs[4], const _Complex float* src)
{

	kb_precompute_gpu(conf->beta);

	//sorting for large problems exceeding shared memory 256 fold (to be debated)
	bool sort = md_calc_size(3, grid_dims) > 256 * 6 * 1024;

	struct grid_plan_s plan = grid_plan_create(*conf, sort, grid_dims, grid_strs, ksp_dims, ksp_strs, trj_strs, traj);

	debug_grid_plan(DP_DEBUG4, &plan);

	if (sort) {

		int grid_size = MIN(1024, plan.access.bin.grid_count);
		dim3 block_size = getBlockSize(1024, (const void*)kern_grid_sorted<false>);
		int mem_size = plan.access.bin.shared_size * sizeof(cuFloatComplex);

		kern_grid_sorted<false><<<grid_size, block_size, mem_size, cuda_get_stream()>>>(plan, (const cuFloatComplex*)traj, (cuFloatComplex*)grid, (const cuFloatComplex*)src);

	} else {

		const long size[3] = { ksp_dims[1], ksp_dims[2], ksp_dims[3] };
		dim3 cu_block = getBlockSize3(size, (const void*)kern_grid<false>);
		dim3 cu_grid = getGridSize3(size, (const void*)kern_grid<false>);

		kern_grid<false><<<cu_grid, cu_block, 0, cuda_get_stream() >>>(plan, (const cuFloatComplex*)traj, (cuFloatComplex*)grid, (const cuFloatComplex*)src);
	}

	grid_plan_free(plan);

	CUDA_KERNEL_ERROR;
}



void cuda_gridH(const struct grid_conf_s* conf, const long ksp_dims[4], const long trj_strs[4], const _Complex float* traj, const long ksp_strs[4], _Complex float* dst, const long grid_dims[4], const long grid_strs[4], const _Complex float* grid)
{

	kb_precompute_gpu(conf->beta);

	//sorting for large problems exceeding shared memory 256 fold (to be debated)
	bool sort = md_calc_size(3, grid_dims) > 256 * 6 * 1024;

	struct grid_plan_s plan = grid_plan_create(*conf, sort, grid_dims, grid_strs, ksp_dims, ksp_strs, trj_strs, traj);

	debug_grid_plan(DP_DEBUG4, &plan);

	if (sort) {

		int grid_size = MIN(1024, plan.access.bin.grid_count);
		dim3 block_size = getBlockSize(1024, (const void*)kern_grid_sorted<false>);
		int mem_size = plan.access.bin.shared_size * sizeof(cuFloatComplex);

		kern_grid_sorted<true><<<grid_size, block_size, mem_size, cuda_get_stream()>>>(plan, (const cuFloatComplex*)traj, (cuFloatComplex*)dst, (const cuFloatComplex*)grid);

	} else {

		const long size[3] = { ksp_dims[1], ksp_dims[2], ksp_dims[3] };
		dim3 cu_block = getBlockSize3(size, (const void*)kern_grid<true>);
		dim3 cu_grid = getGridSize3(size, (const void*)kern_grid<true>);

		kern_grid<true><<<cu_grid, cu_block, 0, cuda_get_stream() >>>(plan, (const cuFloatComplex*)traj, (cuFloatComplex*)dst, (const cuFloatComplex*)grid);
	}

	CUDA_KERNEL_ERROR;

	grid_plan_free(plan);
}


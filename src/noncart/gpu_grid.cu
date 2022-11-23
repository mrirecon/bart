/* Copyright 2022. Uecker Lab. University Medical Center Göttingen.
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

#include "misc/misc.h"

#include "num/multind.h"
#include "num/gpu_misc.h"
#include "num/gpuops.h"
#include "num/multiplace.h"

#include "noncart/grid.h"
#include "gpu_grid.h"

__device__ cuFloatComplex zexp(cuFloatComplex x)
{
	float sc = expf(cuCrealf(x));
	float si;
	float co;
	sincosf(cuCimagf(x), &si, &co);
	return make_cuFloatComplex(sc * co, sc * si);
}

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
				
				cuFloatComplex cval = make_cuFloatComplex(0, val);
				cval = zexp(cval);

				cval.x *= c.scale;
				cval.y *= c.scale;

				if (c.fmac) {

					for (long i = 0; i < c.N; i++)
						dst[idx + i * c.tot] = cuCaddf(dst[idx + i * c.tot], cuCmulf(src[idx + i * c.tot], cval));
				} else {

					for (long i = 0; i < c.N; i++)
						dst[idx + i * c.tot] = cuCmulf(src[idx + i * c.tot], cval);
				}
			}
}



extern "C" void cuda_apply_linphases_3D(int N, const long img_dims[], const float shifts[3], _Complex float* dst, const _Complex float* src, _Bool conj, _Bool fmac, float scale)
{
	struct linphase_conf c;

	c.cn = 0;
	c.tot = 1;
	c.N = 1;
	c.scale = scale;
	c.conj = conj;
	c.fmac = fmac;

	for (int n = 0; n < 3; n++) {

		c.shifts[n] = 2. * M_PI * (float)(shifts[n]) / ((float)img_dims[n]);
		c.cn -= c.shifts[n] * (float)img_dims[n] / 2.;
		
		c.dims[n] = img_dims[n];
		c.tot *= c.dims[n];
	}

	c.N = md_calc_size(N - 3, img_dims + 3);

	const void* func = (const void*)kern_apply_linphases_3D;
	kern_apply_linphases_3D<<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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

static __device__ float posf(int d, int i)
{
	return (1 == d) ? 0. : (((float)i - (float)d / 2.) / (float)d);
}

struct rolloff_conf {

	long dims[3];
	long tot;
	long N;
	float os;
	float width;
	float beta;
	float bessel_beta;
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

				long idx = x + c.dims[0] * (y + c.dims[1] * z);
				
				float val = ((c.dims[0] > 1) ? rolloff(posf(c.dims[0], x) / c.os, c.beta, c.width) * c.bessel_beta : 1)
					  * ((c.dims[1] > 1) ? rolloff(posf(c.dims[1], y) / c.os, c.beta, c.width) * c.bessel_beta : 1)
					  * ((c.dims[2] > 1) ? rolloff(posf(c.dims[2], z) / c.os, c.beta, c.width) * c.bessel_beta : 1);

				for (long i = 0; i < c.N; i++) {

					dst[idx + i * c.tot].x = val * src[idx + i * c.tot].x;
					dst[idx + i * c.tot].y = val * src[idx + i * c.tot].y;
				}
			}
}



extern "C" void cuda_apply_rolloff_correction(float os, float width, float beta, int N, const long dims[], _Complex float* dst, const _Complex float* src)
{
	struct rolloff_conf c = {

		.dims = { dims[0], dims[1], dims[2] },
		.tot = md_calc_size(3, dims),
		.N = md_calc_size(N - 3, dims + 3),
		.os = os,
		.width = width,
		.beta = beta,
		.bessel_beta = bessel_kb_beta,
	};

	const void* func = (const void*)kern_apply_rolloff_correction;
	kern_apply_rolloff_correction<<<getGridSize3(c.dims, func), getBlockSize3(c.dims, (const void*)func), 0, cuda_get_stream()>>>(c, (cuFloatComplex*)dst, (const cuFloatComplex*)src);
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

#define GRID_DIMS 3

struct grid_data {

	float os;
	float width;
	bool periodic;

	long samples;
	long grid_dims[4];
	long ksp_dims[4];

	float shift[GRID_DIMS];

	int ch;
	long off_ch_ksp;
	long off_ch_grid;

	int kb_size;
	float* kb_table;

	long NB;
	long SB_trj;
	long SB_grd;
	long SB_ksp;
};

struct grid_data_device {

	float pos[GRID_DIMS];
	int pos_grid[GRID_DIMS];
	int sti[GRID_DIMS];
	int eni[GRID_DIMS];
	int off[GRID_DIMS];
};

__device__ static __inline__ void dev_atomic_zadd_scl(cuFloatComplex* arg, cuFloatComplex val, float scl)
{
	atomicAdd(&(arg->x), val.x * scl);
	atomicAdd(&(arg->y), val.y * scl);
}

#if 0
__device__ static __inline__ void dev_zadd_scl(cuFloatComplex* arg, cuFloatComplex val, float scl)
{
	arg->x += val.x * scl;
	arg->y += val.y * scl;
}
#endif

__device__ static void grid_point_r(const struct grid_data* gd, const struct grid_data_device* gdd, cuFloatComplex* dst, const cuFloatComplex* src)
{

	float d[GRID_DIMS];
	long ind[GRID_DIMS];

	for (long z = gdd->sti[2]; z <= gdd->eni[2]; z++) {

		d[2] = intlookup(gd->kb_size, gd->kb_table, fabs(((float)z - gdd->pos[2]))/ gd->width);
		ind[2] = ((z + gdd->off[2]) % gd->grid_dims[2]);

		for (long y = gdd->sti[1]; y <= gdd->eni[1]; y++) {

			d[1] = intlookup(gd->kb_size, gd->kb_table, fabs(((float)y - gdd->pos[1]))/ gd->width) * d[2];
			ind[1] = ((y + gdd->off[1]) % gd->grid_dims[1]) + ind[2] * gd->grid_dims[1];

			for (long x = gdd->sti[0]; x <= gdd->eni[0]; x++) {

				d[0] = intlookup(gd->kb_size, gd->kb_table, fabs(((float)x - gdd->pos[0]))/ gd->width) * d[1];
				ind[0] = ((x + gdd->off[0]) % gd->grid_dims[0]) + ind[1] * gd->grid_dims[0];

				dev_atomic_zadd_scl(dst + ind[0] , src[0], d[0]);
			}
		}
	}
}

__device__ static void gridH_point_r(const struct grid_data* gd, const struct grid_data_device* gdd, cuFloatComplex* dst, const cuFloatComplex* src)
{

	float d[GRID_DIMS];
	long ind[GRID_DIMS];

	for (long z = gdd->sti[2]; z <= gdd->eni[2]; z++) {

		d[2] = intlookup(gd->kb_size, gd->kb_table, fabs(((float)z - gdd->pos[2]))/ gd->width);
		ind[2] = ((z + gdd->off[2]) % gd->grid_dims[2]);

		for (long y = gdd->sti[1]; y <= gdd->eni[1]; y++) {

			d[1] = intlookup(gd->kb_size, gd->kb_table, fabs(((float)y - gdd->pos[1]))/ gd->width) * d[2];
			ind[1] = ((y + gdd->off[1]) % gd->grid_dims[1]) + ind[2] * gd->grid_dims[1];

			for (long x = gdd->sti[0]; x <= gdd->eni[0]; x++) {

				d[0] = intlookup(gd->kb_size, gd->kb_table, fabs(((float)x - gdd->pos[0]))/ gd->width) * d[1];
				ind[0] = ((x + gdd->off[0]) % gd->grid_dims[0]) + ind[1] * gd->grid_dims[0];

				dev_atomic_zadd_scl(dst, src[ind[0]], d[0]);
			}
		}
	}
}

__device__ static struct grid_data_device get_grid_data_device(const struct grid_data* conf, const cuFloatComplex traj[GRID_DIMS])
{
	struct grid_data_device gdd;

	for (int j = 0; j < GRID_DIMS; j++) {

		gdd.pos[j] = conf->os * ((traj[j]).x + conf->shift[j]);
		gdd.pos[j] += (conf->grid_dims[j] > 1) ? ((float) conf->grid_dims[j] / 2.) : 0.;

		gdd.sti[j] = (int)ceil(gdd.pos[j] - 0.5 * conf->width);
		gdd.eni[j] = (int)floor(gdd.pos[j] + 0.5 * conf->width);
		gdd.off[j] = 0;

		if (gdd.sti[j] > gdd.eni[j])
			continue;

		if (!conf->periodic) {

			gdd.sti[j] = MAX(gdd.sti[j], 0);
			gdd.eni[j] = MIN(gdd.eni[j], conf->grid_dims[j] - 1);

		} else {

			while (gdd.sti[j] + gdd.off[j] < 0)
				gdd.off[j] += conf->grid_dims[j];
		}

		if (1 == conf->grid_dims[j]) {

			assert(0. == gdd.pos[j]); // ==0. fails nondeterministically for test_nufft_forward bbdec08cb
			gdd.sti[j] = 0;
			gdd.eni[j] = 0;
		}
	}

	return gdd;
}


__global__ static void kern_grid(struct grid_data conf, const cuFloatComplex* traj, cuFloatComplex* grid, const cuFloatComplex* ksp)
{
	int start[3];
	int stride[3];

	start[0] = threadIdx.x + blockDim.x * blockIdx.x;
	start[1] = threadIdx.y + blockDim.y * blockIdx.y;
	start[2] = threadIdx.z + blockDim.z * blockIdx.z;

	stride[0] = blockDim.x * gridDim.x;
	stride[1] = blockDim.y * gridDim.y;
	stride[2] = blockDim.z * gridDim.z;


	struct grid_data_device gdd;

	for (long z = start[2]; z < conf.NB; z +=stride[2])
	for (long c = start[1]; c < conf.ch; c += stride[1])
	for (long i = start[0]; i < conf.samples; i += stride[0]) {

		long offset_trj = z * conf.SB_trj;
		long offset_grd = z * conf.SB_grd + c * conf.off_ch_grid;
		long offset_ksp = z * conf.SB_ksp + c * conf.off_ch_ksp;

		gdd = get_grid_data_device(&conf, traj + offset_trj + i * GRID_DIMS);

		grid_point_r(&conf, &gdd, grid + offset_grd, ksp + i + offset_ksp);
	}
}


void cuda_grid(const struct grid_conf_s* conf, int N, const long traj_dims[], const _Complex float* traj, const long grid_dims[], _Complex float* grid, const long ksp_dims[], const _Complex float* src)
{

	kb_precompute_gpu(conf->beta);

	assert((4 == N) || (5 == N));

	struct grid_data gd = {

		.os = conf->os,
		.width = conf->width,
		.periodic = conf->periodic,

		.samples = ksp_dims[1] * ksp_dims[2],

		.grid_dims = { grid_dims[0], grid_dims[1], grid_dims[2], grid_dims[3]},
		.ksp_dims = { ksp_dims[0], ksp_dims[1], ksp_dims[2], ksp_dims[3]},

		.shift = { conf->shift[0], conf->shift[1], conf->shift[2] },

		.ch = (int)ksp_dims[3],

		.off_ch_ksp = md_calc_size(3, ksp_dims),
		.off_ch_grid = md_calc_size(3, grid_dims),

		.kb_size = kb_size,
		.kb_table = (float*)multiplace_read((struct multiplace_array_s*)kb_table, (const void*)traj),

		.NB = (4 == N) ? 1 : MAX(ksp_dims[4], grid_dims[4]),
		.SB_trj = ((4 == N) || (1 == traj_dims[4])) ? 0 : md_calc_size(4, traj_dims),
		.SB_grd = ((4 == N) || (1 == grid_dims[4])) ? 0 : md_calc_size(4, grid_dims),
		.SB_ksp = ((4 == N) || (1 == ksp_dims[4])) ? 0 : md_calc_size(4, ksp_dims),
	};

	const long size[3] = { gd.samples, gd.ch, gd.NB };
	dim3 cu_block = getBlockSize3(size, (const void*)kern_grid);
	dim3 cu_grid = getGridSize3(size, (const void*)kern_grid);

	kern_grid<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(gd, (const cuFloatComplex*)traj, (cuFloatComplex*)grid, (const cuFloatComplex*)src);

	CUDA_KERNEL_ERROR;
}

__global__ static void kern_gridH(struct grid_data conf, const cuFloatComplex* traj, cuFloatComplex* ksp, const cuFloatComplex* grid)
{
	int start[3];
	int stride[3];

	start[0] = threadIdx.x + blockDim.x * blockIdx.x;
	start[1] = threadIdx.y + blockDim.y * blockIdx.y;
	start[2] = threadIdx.z + blockDim.z * blockIdx.z;

	stride[0] = blockDim.x * gridDim.x;
	stride[1] = blockDim.y * gridDim.y;
	stride[2] = blockDim.z * gridDim.z;


	struct grid_data_device gdd;

	for (long z = start[2]; z < conf.NB; z +=stride[2])
	for (long c = start[1]; c < conf.ch; c += stride[1])
	for (long i = start[0]; i < conf.samples; i += stride[0]) {

		long offset_trj = z * conf.SB_trj;
		long offset_grd = z * conf.SB_grd + c * conf.off_ch_grid;
		long offset_ksp = z * conf.SB_ksp + c * conf.off_ch_ksp;

		gdd = get_grid_data_device(&conf, traj + offset_trj + i * GRID_DIMS);

		gridH_point_r(&conf, &gdd, ksp + i + offset_ksp, grid + offset_grd);
	}
}


void cuda_gridH(const struct grid_conf_s* conf, int N, const long traj_dims[], const _Complex float* traj, const long ksp_dims[], _Complex float* dst, const long grid_dims[], const _Complex float* grid)
{

	kb_precompute_gpu(conf->beta);

	assert((4 == N) || (5 == N));

	struct grid_data gd = {

		.os = conf->os,
		.width = conf->width,
		.periodic = conf->periodic,

		.samples = ksp_dims[1] * ksp_dims[2],

		.grid_dims = { grid_dims[0], grid_dims[1], grid_dims[2], grid_dims[3]},
		.ksp_dims = { ksp_dims[0], ksp_dims[1], ksp_dims[2], ksp_dims[3]},

		.shift = { conf->shift[0], conf->shift[1], conf->shift[2] },

		.ch = (int)ksp_dims[3],

		.off_ch_ksp = md_calc_size(3, ksp_dims),
		.off_ch_grid = md_calc_size(3, grid_dims),

		.kb_size = kb_size,
		.kb_table = (float*)multiplace_read((struct multiplace_array_s*)kb_table, (const void*)traj),

		.NB = (4 == N) ? 1 : MAX(ksp_dims[4], grid_dims[4]),
		.SB_trj = ((4 == N) || (1 == traj_dims[4])) ? 0 : md_calc_size(4, traj_dims),
		.SB_grd = ((4 == N) || (1 == grid_dims[4])) ? 0 : md_calc_size(4, grid_dims),
		.SB_ksp = ((4 == N) || (1 == ksp_dims[4])) ? 0 : md_calc_size(4, ksp_dims),
	};

	const long size[3] = { gd.samples, gd.ch, gd.NB };
	dim3 cu_block = getBlockSize3(size, (const void*)kern_gridH);
	dim3 cu_grid = getGridSize3(size, (const void*)kern_gridH);

	kern_gridH<<<cu_grid, cu_block, 0, cuda_get_stream() >>>(gd, (const cuFloatComplex*)traj, (cuFloatComplex*)dst, (const cuFloatComplex*)grid);

	CUDA_KERNEL_ERROR;
}


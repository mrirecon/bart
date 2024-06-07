#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/gpuops.h"

#include "gpukrnls_triagmat.h"



struct cuda_strides_upper_triagmat {

	long N;	 //Continous vector
	long NC; //Coils dimension (stride N)
	long NM; //Matrix dimension

	//strides of matrix dimension (do not need to be contigous)
	long ostr;
	long istr;
	long mstr;
};

__device__ static long upper_triag_idx(long i, long j)
{
	if (i > j)
		return -(i + ((j + 1) * j) / 2);
	else
		return i + ((j + 1) * j) / 2;
}


__global__ static void kern_zrfmac_upper_triagmat(struct cuda_strides_upper_triagmat strs, float2* dst, const float2* src, const float* mat)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < strs.N; i += stride) {

		for (long m = 0; m < strs.NM; m++) {

			for (long n = 0; n <= m; n++) {

				float val = mat[i + upper_triag_idx(n, m) * strs.mstr];

				if (0. == val)
					continue;

				for (long c = 0; c < strs.NC; c++) {

					dst[i + strs.N * c + strs.ostr * m].x += val * src[i + strs.N * c + strs.istr * n].x;
					dst[i + strs.N * c + strs.ostr * m].y += val * src[i + strs.N * c + strs.istr * n].y;

					if (n != m) {

						dst[i + strs.N * c + strs.ostr * n].x += val * src[i + strs.N * c + strs.istr * m].x;
						dst[i + strs.N * c + strs.ostr * n].y += val * src[i + strs.N * c + strs.istr * m].y;
					}
				}
			}
		}
	}
}

#define BLOCKSIZE 1024

static int blocksize(long N)
{
	return BLOCKSIZE;
}

static long gridsize(long N)
{
	// to ensure that "start" does not overflow we need to restrict gridsize!
	return MIN((N + BLOCKSIZE - 1) / BLOCKSIZE, 65536 - 1);
}


extern "C" void cuda_zrfmac_upper_triagmat(long N, long NC, long NM, long ostr, long istr, long mstr, float* dst, const float* src, const float* mat)
{
	cuda_strides_upper_triagmat conf;
	conf.N = N;
	conf.NC = NC;
	conf.NM = NM;
	conf.ostr = ostr / 2; //use float2 for faster access
	conf.istr = istr / 2; //use float2 for faster access
	conf.mstr = mstr;

	kern_zrfmac_upper_triagmat<<<gridsize(N), blocksize(N), 0, cuda_get_stream()>>>(conf, (float2*)dst, (float2*)src, mat);
	CUDA_KERNEL_ERROR;
}

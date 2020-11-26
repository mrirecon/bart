#include <cstdint>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

//#define USE_FXDIV
#ifdef USE_FXDIV
#include "num/fxdiv.h"
#endif
#include "num/gpu_conv.h"
#include "num/multind.h"

// limited by hardware to 1024 on most devices
// should be a multiple of 32 (warp size)
#define BLOCKSIZE 1024

static int blocksize(int N)
{
	return BLOCKSIZE;
}

static long gridsize(long N)
{
	return (N + BLOCKSIZE - 1) / BLOCKSIZE;
}

#ifdef USE_FXDIV
__device__ static inline uint32_t cuda_fxdiv_quotient_uint32_t(uint32_t n, const struct fxdiv_divisor_uint32_t divisor) {

	const uint32_t t = __umulhi(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}

__device__ static inline uint64_t cuda_fxdiv_quotient_uint64_t(uint64_t n, const struct fxdiv_divisor_uint64_t divisor) {

	if (1 == divisor.m && 0 == divisor.s1 && 0 == divisor.s2)
		return n;

	const uint64_t t = __umul64hi(n, divisor.m);
	return (t + ((n - t) >> divisor.s1)) >> divisor.s2;
}
#endif


#define MAX_CONV_DIMS 3
struct im2col_descriptor_uint64 {

	uint64_t NC;		// number channels
	long istrs_NC;		// 1
	long ostrs_NC;		// 1

	int N_conv_dims;	// number convolutions (i.e. 1D, 2D, 3D)

	uint64_t odims[MAX_CONV_DIMS];	// dimensions of the convolution (not including channel)
	uint64_t kdims[MAX_CONV_DIMS];
	uint64_t idims[MAX_CONV_DIMS];

	long istrs_odims[MAX_CONV_DIMS];	// input strides of im2col (in elements)
	long istrs_kdims[MAX_CONV_DIMS];
	long ostrs_kdims[MAX_CONV_DIMS];	// output strides of im2col (in elements)
	long ostrs_odims[MAX_CONV_DIMS];

	#ifdef USE_FXDIV
		struct fxdiv_divisor_uint64_t div_NC;			//efficient fixed divisor for dimensions
		struct fxdiv_divisor_uint64_t div_odims[MAX_CONV_DIMS];
		struct fxdiv_divisor_uint64_t div_kdims[MAX_CONV_DIMS];
		struct fxdiv_divisor_uint64_t div_idims[MAX_CONV_DIMS];
	#endif

	long N_in_elements;		// channels * in-dims
	long N_out_elements;		// channels * out-dims * krn-dims
	long N_out_elements_o_only;	// channels * out-dims
	long N_out_elements_k_only;	// channels * krn-dims

	bool triv_strides_dilation;	// trivial dilation and strides
};

// same struct as im2col_descriptor_uint64 but with uint32_t for faster divisons
struct im2col_descriptor_uint32 {

	uint32_t NC;		// number channels
	long istrs_NC;		// 1
	long ostrs_NC;		// 1

	int N_conv_dims;	// number convolutions (i.e. 1D, 2D, 3D)

	uint32_t odims[MAX_CONV_DIMS];	// dimensions of the convolution (not including channel)
	uint32_t kdims[MAX_CONV_DIMS];
	uint32_t idims[MAX_CONV_DIMS];

	long istrs_odims[MAX_CONV_DIMS];	// input strides of im2col (in elements)
	long istrs_kdims[MAX_CONV_DIMS];
	long ostrs_kdims[MAX_CONV_DIMS];	// output strides of im2col (in elements)
	long ostrs_odims[MAX_CONV_DIMS];

	#ifdef USE_FXDIV
		struct fxdiv_divisor_uint32_t div_NC;			//efficient fixed divisor for dimensions
		struct fxdiv_divisor_uint32_t div_odims[MAX_CONV_DIMS];
		struct fxdiv_divisor_uint32_t div_kdims[MAX_CONV_DIMS];
		struct fxdiv_divisor_uint32_t div_idims[MAX_CONV_DIMS];
	#endif

	long N_in_elements;		// channels * in-dims
	long N_out_elements;		// channels * out-dims * krn-dims
	long N_out_elements_o_only;	// channels * out-dims
	long N_out_elements_k_only;	// channels * krn-dims

	bool triv_strides_dilation;	// trivial dilation and strides
};


static struct im2col_descriptor_uint64 get_im2col_descriptor_uint64(const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	struct im2col_descriptor_uint64 config;

	config.NC = idims[1];
	config.istrs_NC = 1;
	config.ostrs_NC = 1;
	#ifdef USE_FXDIV
		config.div_NC = fxdiv_init_uint64_t(idims[1]);
	#endif
	config.N_conv_dims = 0;
	config.N_in_elements = idims[1];
	config.N_out_elements = idims[1];
	config.N_out_elements_o_only = idims[1];
	config.N_out_elements_k_only = idims[1];

	config.triv_strides_dilation = true;

	long istrs[5];
	md_calc_strides(5, istrs, idims, 1);

	for (int i = 0; i < MAX_CONV_DIMS; i++) {

		config.odims[i] = 1;
		config.kdims[i] = 1;
		config.idims[i] = 1;
		config.istrs_odims[i] = 0;
		config.istrs_kdims[i] = 0;
		config.ostrs_odims[i] = 0;
		config.ostrs_kdims[i] = 0;

		#ifdef USE_FXDIV
			config.div_odims[i] = fxdiv_init_uint64_t(1);
			config.div_kdims[i] = fxdiv_init_uint64_t(1);
			config.div_idims[i] = fxdiv_init_uint64_t(1);
		#endif
	}


	for (int i = 2, j = 0; i < 5; i++) {

		if ((1 < odims[i] || 1 < kdims[i]))
			config.N_conv_dims++;
		else
		 	continue;

		config.odims[j] = odims[i];
		config.kdims[j] = kdims[i];
		config.idims[j] = idims[i];

		config.istrs_odims[j] = istrs[i] * (NULL == strides ? 1 : strides[i]);
		config.istrs_kdims[j] = istrs[i] * (NULL == dilation ? 1 : dilation[i]);

		#ifdef USE_FXDIV
			config.div_odims[j] = fxdiv_init_uint64_t(odims[i]);
			config.div_kdims[j] = fxdiv_init_uint64_t(kdims[i]);
			config.div_idims[j] = fxdiv_init_uint64_t(idims[i]);
		#endif

		config.N_in_elements *= idims[i];
		config.N_out_elements_o_only *= odims[i];
		config.N_out_elements_k_only *= kdims[i];
		config.N_out_elements *= odims[i] * kdims[i];

		config.triv_strides_dilation &= ((config.istrs_odims[j] == istrs[i]) && (config.istrs_kdims[j] == istrs[i]));
		j++;
	};

	config.ostrs_odims[0] = config.N_out_elements_k_only;
	config.ostrs_kdims[0] = config.NC;

	for (int i = 1; i < config.N_conv_dims; i++) {

		config.ostrs_odims[i] = config.ostrs_odims[i - 1] * config.odims[i - 1];
		config.ostrs_kdims[i] = config.ostrs_kdims[i - 1] * config.kdims[i - 1];
	}

	return config;
}

static struct im2col_descriptor_uint32 get_im2col_descriptor_uint32(const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	struct im2col_descriptor_uint32 config;

	config.NC = idims[1];
	config.istrs_NC = 1;
	config.ostrs_NC = 1;
	#ifdef USE_FXDIV
		config.div_NC = fxdiv_init_uint32_t(idims[1]);
	#endif
	config.N_conv_dims = 0;
	config.N_in_elements = idims[1];
	config.N_out_elements = idims[1];
	config.N_out_elements_o_only = idims[1];
	config.N_out_elements_k_only = idims[1];

	config.triv_strides_dilation = true;

	long istrs[5];
	md_calc_strides(5, istrs, idims, 1);

	for (int i = 0; i < MAX_CONV_DIMS; i++) {

		config.odims[i] = 1;
		config.kdims[i] = 1;
		config.idims[i] = 1;
		config.istrs_odims[i] = 0;
		config.istrs_kdims[i] = 0;
		config.ostrs_odims[i] = 0;
		config.ostrs_kdims[i] = 0;

		#ifdef USE_FXDIV
			config.div_odims[i] = fxdiv_init_uint32_t(1);
			config.div_kdims[i] = fxdiv_init_uint32_t(1);
			config.div_idims[i] = fxdiv_init_uint32_t(1);
		#endif
	}


	for (int i = 2, j = 0; i < 5; i++) {

		if ((1 < odims[i] || 1 < kdims[i]))
			config.N_conv_dims++;
		else
		 	continue;

		config.odims[j] = odims[i];
		config.kdims[j] = kdims[i];
		config.idims[j] = idims[i];

		config.istrs_odims[j] = istrs[i] * (NULL == strides ? 1 : strides[i]);
		config.istrs_kdims[j] = istrs[i] * (NULL == dilation ? 1 : dilation[i]);

		#ifdef USE_FXDIV
			config.div_odims[j] = fxdiv_init_uint32_t(odims[i]);
			config.div_kdims[j] = fxdiv_init_uint32_t(kdims[i]);
			config.div_idims[j] = fxdiv_init_uint32_t(idims[i]);
		#endif

		config.N_in_elements *= idims[i];
		config.N_out_elements_o_only *= odims[i];
		config.N_out_elements_k_only *= kdims[i];
		config.N_out_elements *= odims[i] * kdims[i];

		config.triv_strides_dilation &= ((config.istrs_odims[j] == istrs[i]) && (config.istrs_kdims[j] == istrs[i]));
		j++;
	};

	config.ostrs_odims[0] = config.N_out_elements_k_only;
	config.ostrs_kdims[0] = config.NC;

	for (int i = 1; i < config.N_conv_dims; i++) {

		config.ostrs_odims[i] = config.ostrs_odims[i - 1] * config.odims[i - 1];
		config.ostrs_kdims[i] = config.ostrs_kdims[i - 1] * config.kdims[i - 1];
	}

	return config;
}

// loop over out-dims and krn-dims and copy elements from input (copies one element per thread)
__global__ static void kern_im2col_valid(struct im2col_descriptor_uint64 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_out_elements; i += stride) {

		uint64_t i_cur = i;
		uint64_t i_new = i;
		long in_index = 0;

		if (1 < config.NC) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_NC);
			#else
				i_new = i_cur / config.NC;
			#endif
			in_index = (i_cur - config.NC * i_new) * config.istrs_NC;
			i_cur = i_new;
		}

		for (int j = 0; j < config.N_conv_dims; j++) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_kdims[j]);
			#else
				i_new = i_cur / config.kdims[j];
			#endif
			in_index += config.istrs_kdims[j] * (i_cur - config.kdims[j] * i_new);
			i_cur = i_new;
		}

		for (int j = 0; j < config.N_conv_dims - 1; j++) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_odims[j]);
			#else
				i_new = i_cur / config.odims[j];
			#endif
			in_index += config.istrs_odims[j] * (i_cur - config.odims[j] * i_new);
			i_cur = i_new;
		}

		in_index += i_cur * config.istrs_odims[config.N_conv_dims - 1];

		dst[i] = src[in_index];
	}
}

// loop over out-dims and krn-dims and copy elements from input (copies one element per thread)
__global__ static void kern_im2col_valid(struct im2col_descriptor_uint32 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_out_elements; i += stride) {


		uint32_t i_cur = i;
		uint32_t i_new = i;
		long in_index = 0;

		if (1 < config.NC) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_NC);
			#else
				i_new = i_cur / config.NC;
			#endif
			in_index = (i_cur - config.NC * i_new) * config.istrs_NC;
			i_cur = i_new;
		}

		for (int j = 0; j < config.N_conv_dims; j++) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_kdims[j]);
			#else
				i_new = i_cur / config.kdims[j];
			#endif
			in_index += config.istrs_kdims[j] * (i_cur - config.kdims[j] * i_new);
			i_cur = i_new;
		}

		for (int j = 0; j < config.N_conv_dims - 1; j++) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_odims[j]);
			#else
				i_new = i_cur / config.odims[j];
			#endif
			in_index += config.istrs_odims[j] * (i_cur - config.odims[j] * i_new);
			i_cur = i_new;
		}

		in_index += i_cur * config.istrs_odims[config.N_conv_dims - 1];

		dst[i] = src[in_index];
	}
}

// loop over in-dims and copy elements from input to all corresponding output position
__global__ static void kern_im2col_valid_no_dil_str(struct im2col_descriptor_uint64 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_in_elements; i += stride) {

		#ifdef USE_FXDIV
			uint64_t i_cur = i;
			uint64_t i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_NC);
			uint64_t c = i_cur - i_new * config.NC;
			i_cur = i_new;

			i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_idims[0]);
			uint64_t ix = i_cur - i_new * config.idims[0];
			i_cur = i_new;

			i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_idims[1]);

			uint64_t iy = i_cur - i_new * config.idims[1];
			uint64_t iz = i_new;
		#else
			uint64_t i_cur = i;
			uint64_t i_new = i_cur / config.NC;
			uint64_t c = i_cur - i_new * config.NC;
			i_cur = i_new;

			i_new = i_cur / config.idims[0];
			uint64_t ix = i_cur - i_new * config.idims[0];
			i_cur = i_new;

			i_new = i_cur / config.idims[1];
			uint64_t iy = i_cur - i_new * config.idims[1];
			uint64_t iz = i_new;
		#endif

		cuFloatComplex tmp = src[i];

		for(uint kz = 0; kz < config.kdims[2]; kz++)
		for(uint ky = 0; ky < config.kdims[1]; ky++)
		for(uint kx = 0; kx < config.kdims[0]; kx++) {

			int oz = iz - kz;
			int oy = iy - ky;
			int ox = ix - kx;

			long offset_z = config.N_out_elements_k_only * config.odims[0] * config.odims[1] * oz + config.NC * config.kdims[0] * config.kdims[1] * kz;
			long offset_y = config.N_out_elements_k_only * config.odims[0] * oy + config.NC * config.kdims[0] * ky;
			long offset_x = config.N_out_elements_k_only * ox + config.NC * kx;

			long index = c 	+ offset_x + offset_y + offset_z;

			if (   ((0 <= ox) && ((int)config.odims[0] > ox))
			    && ((0 <= oy) && ((int)config.odims[1] > oy))
			    && ((0 <= oz) && ((int)config.odims[2] > oz)))
				dst[index] = tmp;

		}
	}
}

// loop over in-dims and copy elements from input to all corresponding output position
__global__ static void kern_im2col_valid_no_dil_str(struct im2col_descriptor_uint32 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_in_elements; i += stride) {

		#ifdef USE_FXDIV
			uint32_t i_cur = i;
			uint32_t i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_NC);
			uint32_t c = i_cur - i_new * config.NC;
			i_cur = i_new;

			i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_idims[0]);
			uint32_t ix = i_cur - i_new * config.idims[0];
			i_cur = i_new;

			i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_idims[1]);

			uint32_t iy = i_cur - i_new * config.idims[1];
			uint32_t iz = i_new;
		#else
			uint32_t i_cur = i;
			uint32_t i_new = i_cur / config.NC;
			uint32_t c = i_cur - i_new * config.NC;
			i_cur = i_new;

			i_new = i_cur / config.idims[0];
			uint32_t ix = i_cur - i_new * config.idims[0];
			i_cur = i_new;

			i_new = i_cur / config.idims[1];
			uint32_t iy = i_cur - i_new * config.idims[1];
			uint32_t iz = i_new;
		#endif

		cuFloatComplex tmp = src[i];

		for(uint kz = 0; kz < config.kdims[2]; kz++)
		for(uint ky = 0; ky < config.kdims[1]; ky++)
		for(uint kx = 0; kx < config.kdims[0]; kx++) {

			int oz = iz - kz;
			int oy = iy - ky;
			int ox = ix - kx;

			long offset_z = config.N_out_elements_k_only * config.odims[0] * config.odims[1] * oz + config.NC * config.kdims[0] * config.kdims[1] * kz;
			long offset_y = config.N_out_elements_k_only * config.odims[0] * oy + config.NC * config.kdims[0] * ky;
			long offset_x = config.N_out_elements_k_only * ox + config.NC * kx;

			long index = c 	+ offset_x + offset_y + offset_z;

			if (   ((0 <= ox) && ((int)config.odims[0] > ox))
			    && ((0 <= oy) && ((int)config.odims[1] > oy))
			    && ((0 <= oz) && ((int)config.odims[2] > oz)))
				dst[index] = tmp;

		}
	}
}

/* *
 * Optimized kernel for copying im2col (complex float only)
 *
 * @args dst
 * @args src
 * @args odims		[OC,  1, OX, OY, OZ]
 * @args idims		[ 1, IC, IX, IY, IZ]
 * @args kdims		[OC, IC, KX, KY, KZ]
 * @args dilation	[ 1,  1, DX, DY, DZ] or NULL
 * @args strides	[ 1,  1, SX, SY, SZ] or NULL
 *
 * Copy:
 * dims:	[IC, KX, KY, KZ, OX, OY, OZ]
 * ostrs:	trivial strides of dims
 * istrs:	[ISC, ISX * DX, ISY * DY, ISZ * DZ, ISX * SX, ISY * SY, ISZ * SZ]
 * where IS* are trivial strides of idims
 * */
extern "C" void cuda_im2col(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{
	struct im2col_descriptor_uint64 config64 = get_im2col_descriptor_uint64(odims, idims, kdims, dilation, strides);
	struct im2col_descriptor_uint32 config32 = get_im2col_descriptor_uint32(odims, idims, kdims, dilation, strides);

	if ((1 < config32.NC) && (config64.triv_strides_dilation)) {

		if (config32.N_out_elements < INT32_MAX)
			kern_im2col_valid_no_dil_str<<<gridsize(config32.N_in_elements), blocksize(config32.N_in_elements)>>>(config32, (cuFloatComplex*) dst, (cuFloatComplex*) src);
		else
	 		kern_im2col_valid_no_dil_str<<<gridsize(config64.N_in_elements), blocksize(config32.N_in_elements)>>>(config64, (cuFloatComplex*) dst, (cuFloatComplex*) src);
	} else {

		if (config32.N_out_elements < INT32_MAX)
			kern_im2col_valid<<<gridsize(config32.N_out_elements), blocksize(config32.N_out_elements)>>>(config32, (cuFloatComplex*) dst, (cuFloatComplex*) src);
		else
			kern_im2col_valid<<<gridsize(config64.N_out_elements), blocksize(config64.N_out_elements)>>>(config64, (cuFloatComplex*) dst, (cuFloatComplex*) src);

	}
}

__global__ static void kern_im2col_valid_no_dil_str_transp(struct im2col_descriptor_uint64 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_in_elements; i += stride) {

		uint64_t i_cur = i;
		uint64_t i_new;

	#ifdef USE_FXDIV
		i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_NC);
	#else
		i_new = i_cur / config.NC;
	#endif
		uint64_t c = i_cur - i_new * config.NC;
		i_cur = i_new;

	#ifdef USE_FXDIV
		i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_idims[0]);
	#else
		i_new = i_cur / config.idims[0];
	#endif
		uint64_t ix = i_cur - i_new * config.idims[0];
		i_cur = i_new;

	#ifdef USE_FXDIV
		i_new = cuda_fxdiv_quotient_uint64_t(i_cur, config.div_idims[1]);
	#else
		i_new = i_cur / config.idims[1];
	#endif
		uint64_t iy = i_cur - i_new * config.idims[1];
		uint64_t iz = i_new;

		cuFloatComplex result = dst[i];

		for(uint kz = 0; kz < config.kdims[2]; kz++) {

			int oz = iz - kz;
			if ((0 > oz) || ((int)config.odims[2] <= oz))
				continue;

			long offset_z = config.N_out_elements_k_only * config.odims[0] * config.odims[1] * oz + config.NC * config.kdims[0] * config.kdims[1] * kz;

			for(uint ky = 0; ky < config.kdims[1]; ky++) {

				int oy = iy - ky;
				if ((0 > oy) || ((int)config.odims[1] <= oy))
					continue;

				long offset_y = config.N_out_elements_k_only * config.odims[0] * oy + config.NC * config.kdims[0] * ky;

				for(uint kx = 0; kx < config.kdims[0]; kx++) {

					int ox = ix - kx;

					long offset_x = config.N_out_elements_k_only * ox + config.NC * kx;

					long index = c 	+ offset_x + offset_y + offset_z;

					if ((0 <= ox) && ((int)config.odims[0] > ox))
						result = cuCaddf(result, src[index]);
				}
			}
		}
		dst[i] = result;
	}
}

__global__ static void kern_im2col_valid_no_dil_str_transp(struct im2col_descriptor_uint32 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_in_elements; i += stride) {

		uint32_t i_cur = i;
		uint32_t i_new;

	#ifdef USE_FXDIV
		i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_NC);
	#else
		i_new = i_cur / config.NC;
	#endif
		uint32_t c = i_cur - i_new * config.NC;
		i_cur = i_new;

	#ifdef USE_FXDIV
		i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_idims[0]);
	#else
		i_new = i_cur / config.idims[0];
	#endif
		uint32_t ix = i_cur - i_new * config.idims[0];
		i_cur = i_new;

	#ifdef USE_FXDIV
		i_new = cuda_fxdiv_quotient_uint32_t(i_cur, config.div_idims[1]);
	#else
		i_new = i_cur / config.idims[1];
	#endif
		uint32_t iy = i_cur - i_new * config.idims[1];
		uint32_t iz = i_new;

		cuFloatComplex result = dst[i];

		for(uint kz = 0; kz < config.kdims[2]; kz++) {

			int oz = iz - kz;
			if ((0 > oz) || ((int)config.odims[2] <= oz))
				continue;

			long offset_z = config.N_out_elements_k_only * config.odims[0] * config.odims[1] * oz + config.NC * config.kdims[0] * config.kdims[1] * kz;

			for(uint ky = 0; ky < config.kdims[1]; ky++) {

				int oy = iy - ky;
				if ((0 > oy) || ((int)config.odims[1] <= oy))
					continue;

				long offset_y = config.N_out_elements_k_only * config.odims[0] * oy + config.NC * config.kdims[0] * ky;

				for(uint kx = 0; kx < config.kdims[0]; kx++) {

					int ox = ix - kx;

					long offset_x = config.N_out_elements_k_only * ox + config.NC * kx;

					long index = c 	+ offset_x + offset_y + offset_z;

					if ((0 <= ox) && ((int)config.odims[0] > ox))
						result = cuCaddf(result, src[index]);
				}
			}
		}
		dst[i] = result;
	}
}

#ifdef NON_DETERMINISTIC

__global__ static void kern_im2col_valid_transp(struct im2col_descriptor_uint32 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_out_elements; i += stride) {


		uint32_t i0 = i;
		uint32_t i_new = i;
		long in_index = 0;

		if (1 < config.NC) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint32_t(i0, config.div_NC);
			#else
				i_new = i0 / config.NC;
			#endif
			in_index = (i0 - config.NC * i_new) * config.istrs_NC;
			i0 = i_new;
		}

		for (int j = 0; j < config.N_conv_dims; j++) {

		#ifdef USE_FXDIV
			i_new = cuda_fxdiv_quotient_uint32_t(i0, config.div_kdims[j]);
		#else
			i_new = i0 / config.kdims[j];
		#endif
			in_index += config.istrs_kdims[j] * (i0 - config.kdims[j] * i_new);
			i0 = i_new;
		}

		for (int j = 0; j < config.N_conv_dims - 1; j++) {

		#ifdef USE_FXDIV
			i_new = cuda_fxdiv_quotient_uint32_t(i0, config.div_odims[j]);
		#else
			i_new = i0 / config.odims[j];
		#endif
			in_index += config.istrs_odims[j] * (i0 - config.odims[j] * i_new);
			i0 = i_new;
		}

		in_index += i0 * config.istrs_odims[config.N_conv_dims - 1];

		atomicAdd(&(dst[in_index].x), src[i].x);
		atomicAdd(&(dst[in_index].y), src[i].y);
	}
}

__global__ static void kern_im2col_valid_transp(struct im2col_descriptor_uint64 config, cuFloatComplex* dst, const cuFloatComplex* src)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (long i = start; i < config.N_out_elements; i += stride) {


		uint64_t i0 = i;
		uint64_t i_new = i;
		long in_index = 0;

		if (1 < config.NC) {

			#ifdef USE_FXDIV
				i_new = cuda_fxdiv_quotient_uint64_t(i0, config.div_NC);
			#else
				i_new = i0 / config.NC;
			#endif
			in_index = (i0 - config.NC * i_new) * config.istrs_NC;
			i0 = i_new;
		}

		for (int j = 0; j < config.N_conv_dims; j++) {

		#ifdef USE_FXDIV
			i_new = cuda_fxdiv_quotient_uint64_t(i0, config.div_kdims[j]);
		#else
			i_new = i0 / config.kdims[j];
		#endif
			in_index += config.istrs_kdims[j] * (i0 - config.kdims[j] * i_new);
			i0 = i_new;
		}

		for (int j = 0; j < config.N_conv_dims - 1; j++) {

		#ifdef USE_FXDIV
			i_new = cuda_fxdiv_quotient_uint64_t(i0, config.div_odims[j]);
		#else
			i_new = i0 / config.odims[j];
		#endif
			in_index += config.istrs_odims[j] * (i0 - config.odims[j] * i_new);
			i0 = i_new;
		}

		in_index += i0 * config.istrs_odims[config.N_conv_dims - 1];

		atomicAdd(&(dst[in_index].x), src[i].x);
		atomicAdd(&(dst[in_index].y), src[i].y);
	}
}

#endif

/* *
 * Transposed/adjoint of cuda im2col
 *
 * @args dst
 * @args src
 * @args odims		[OC,  1, OX, OY, OZ]
 * @args idims		[ 1, IC, IX, IY, IZ]
 * @args kdims		[OC, IC, KX, KY, KZ]
 * @args dilation	[ 1,  1, DX, DY, DZ] or NULL
 * @args strides	[ 1,  1, SX, SY, SZ] or NULL
 *
 * zadd with strides:
 * dims:	[IC, KX, KY, KZ, OX, OY, OZ]
 * ostrs:	[ISC, ISX * DX, ISY * DY, ISZ * DZ, ISX * SX, ISY * SY, ISZ * SZ]
 * istrs:	trivial strides of dims
 * where IS* are trivial strides of idims
 * */
extern "C" void cuda_im2col_transp(_Complex float* dst, const _Complex float* src, const long odims[5], const long idims[5], const long kdims[5], const long dilation[5], const long strides[5])
{

	struct im2col_descriptor_uint64 config64 = get_im2col_descriptor_uint64(odims, idims, kdims, dilation, strides);
	struct im2col_descriptor_uint32 config32 = get_im2col_descriptor_uint32(odims, idims, kdims, dilation, strides);

#ifdef NON_DETERMINISTIC

	if ((1 < config32.NC) && (config64.triv_strides_dilation)) {

		if (config32.N_out_elements < INT32_MAX)
			kern_im2col_valid_no_dil_str_transp<<<gridsize(config32.N_in_elements), blocksize(config32.N_in_elements)>>>(config32, (cuFloatComplex*) dst, (cuFloatComplex*) src);
		else
	 		kern_im2col_valid_no_dil_str_transp<<<gridsize(config64.N_in_elements), blocksize(config32.N_in_elements)>>>(config64, (cuFloatComplex*) dst, (cuFloatComplex*) src);
	} else {

		if (config32.N_out_elements < INT32_MAX)
			kern_im2col_valid_transp<<<gridsize(config32.N_out_elements), blocksize(config32.N_out_elements)>>>(config32, (cuFloatComplex*) dst, (cuFloatComplex*) src);
		else
			kern_im2col_valid_transp<<<gridsize(config64.N_out_elements), blocksize(config64.N_out_elements)>>>(config64, (cuFloatComplex*) dst, (cuFloatComplex*) src);
	}

#else
	assert(config64.triv_strides_dilation);

	if (config32.N_out_elements < INT32_MAX)
		kern_im2col_valid_no_dil_str_transp<<<gridsize(config32.N_in_elements), blocksize(config32.N_in_elements)>>>(config32, (cuFloatComplex*) dst, (cuFloatComplex*) src);
	else
 		kern_im2col_valid_no_dil_str_transp<<<gridsize(config64.N_in_elements), blocksize(config32.N_in_elements)>>>(config64, (cuFloatComplex*) dst, (cuFloatComplex*) src);

#endif
}
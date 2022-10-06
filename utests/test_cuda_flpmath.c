#include <complex.h>

#include "num/flpmath.h"
#include "num/multind.h"
#include "num/rand.h"
#include "num/init.h"

#include "misc/misc.h"
#include "misc/debug.h"

#include "utest.h"

typedef void (*md_2op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1);
typedef void (*md_z2op_t)(int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1);
typedef void (*md_2opf_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const double* iptr1);
typedef void (*md_2opd_t)(int D, const long dims[D], const long ostrs[D], double* optr, const long istrs1[D], const float* iptr1);
typedef void (*md_z2opf_t)(int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex double* iptr1);
typedef void (*md_z2opd_t)(int D, const long dims[D], const long ostrs[D], complex double* optr, const long istrs1[D], const complex float* iptr1);


typedef void (*md_3op_t)(int D, const long dims[D], const long ostrs[D], float* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);
typedef void (*md_z3op_t)(int D, const long dims[D], const long ostrs[D], complex float* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);
typedef void (*md_3opd_t)(int D, const long dims[D], const long ostrs[D], double* optr, const long istrs1[D], const float* iptr1, const long istrs2[D], const float* iptr2);
typedef void (*md_z3opd_t)(int D, const long dims[D], const long ostrs[D], complex double* optr, const long istrs1[D], const complex float* iptr1, const long istrs2[D], const complex float* iptr2);




static bool test_md_z3op(md_z3op_t function)
{
	num_init_gpu();

	enum { N = 3 };
	const long dims[N] = { 2, 5, 3 };
	long strs[N];

	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* optr_cpu = md_alloc(N, dims, CFL_SIZE);
	complex float* iptr1_cpu = md_alloc(N, dims, CFL_SIZE);
	complex float* iptr2_cpu = md_alloc(N, dims, CFL_SIZE);

	complex float* optr_gpu_cpu = md_alloc(N, dims, CFL_SIZE);

	complex float* optr_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
	complex float* iptr1_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
	complex float* iptr2_gpu = md_alloc_gpu(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, optr_cpu);
	md_gaussian_rand(N, dims, iptr1_cpu);
	md_gaussian_rand(N, dims, iptr2_cpu);

	md_copy(N, dims, optr_gpu, optr_cpu, CFL_SIZE);
	md_copy(N, dims, iptr1_gpu, iptr1_cpu, CFL_SIZE);
	md_copy(N, dims, iptr2_gpu, iptr2_cpu, CFL_SIZE);

	function(N, dims, strs, optr_gpu, strs, iptr1_gpu, strs, iptr2_gpu);
	function(N, dims, strs, optr_cpu, strs, iptr1_cpu, strs, iptr2_cpu);

	md_copy(N, dims, optr_gpu_cpu, optr_gpu, CFL_SIZE);

	float err = md_znrmse(N, dims, optr_cpu, optr_gpu_cpu);

	md_free(optr_cpu);
	md_free(iptr1_cpu);
	md_free(iptr2_cpu);

	md_free(optr_gpu_cpu);

	md_free(optr_gpu);
	md_free(iptr1_gpu);
	md_free(iptr2_gpu);

	if ((UT_TOL < err) || (!safe_isfinite(err)))
		debug_printf(DP_WARN, "err: %e\n", err);

	return (UT_TOL >= err) && safe_isfinite(err);
}

static bool test_md_zrmul2(void) { UT_ASSERT(test_md_z3op(md_zrmul2));}
UT_GPU_REGISTER_TEST(test_md_zrmul2);

static bool test_md_zmul2(void) { UT_ASSERT(test_md_z3op(md_zmul2));}
UT_GPU_REGISTER_TEST(test_md_zmul2);

static bool test_md_zdiv2(void) { UT_ASSERT(test_md_z3op(md_zdiv2));}
UT_GPU_REGISTER_TEST(test_md_zdiv2);

static bool test_md_zmulc2(void) { UT_ASSERT(test_md_z3op(md_zmulc2));}
UT_GPU_REGISTER_TEST(test_md_zmulc2);

static bool test_md_zpow2(void) { UT_ASSERT(test_md_z3op(md_zpow2));}
UT_GPU_REGISTER_TEST(test_md_zpow2);

static bool test_md_zfmac2(void) { UT_ASSERT(test_md_z3op(md_zfmac2));}
UT_GPU_REGISTER_TEST(test_md_zfmac2);

static bool test_md_zfmacc2(void) { UT_ASSERT(test_md_z3op(md_zfmacc2));}
UT_GPU_REGISTER_TEST(test_md_zfmacc2);

static bool test_md_ztenmul2(void) { UT_ASSERT(test_md_z3op(md_ztenmul2));}
UT_GPU_REGISTER_TEST(test_md_ztenmul2);

static bool test_md_ztenmulc2(void) { UT_ASSERT(test_md_z3op(md_ztenmulc2));}
UT_GPU_REGISTER_TEST(test_md_ztenmulc2);

static bool test_md_zadd2(void) { UT_ASSERT(test_md_z3op(md_zadd2));}
UT_GPU_REGISTER_TEST(test_md_zadd2);

static bool test_md_zsub2(void) { UT_ASSERT(test_md_z3op(md_zsub2));}
UT_GPU_REGISTER_TEST(test_md_zsub2);

static bool test_md_zmax2(void) { UT_ASSERT(test_md_z3op(md_zmax2));}
UT_GPU_REGISTER_TEST(test_md_zmax2);

static bool test_md_zlessequal2(void) { UT_ASSERT(test_md_z3op(md_zlessequal2));}
UT_GPU_REGISTER_TEST(test_md_zlessequal2);

static bool test_md_zgreatequal2(void) { UT_ASSERT(test_md_z3op(md_zgreatequal2));}
UT_GPU_REGISTER_TEST(test_md_zgreatequal2);




static bool test_md_z2op(md_z2op_t function)
{
	num_init_gpu();

	enum { N = 3 };
	const long dims[N] = { 2, 5, 3 };
	long strs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);

	complex float* optr_cpu = md_alloc(N, dims, CFL_SIZE);
	complex float* iptr1_cpu = md_alloc(N, dims, CFL_SIZE);

	complex float* optr_gpu_cpu = md_alloc(N, dims, CFL_SIZE);

	complex float* optr_gpu = md_alloc_gpu(N, dims, CFL_SIZE);
	complex float* iptr1_gpu = md_alloc_gpu(N, dims, CFL_SIZE);

	md_gaussian_rand(N, dims, optr_cpu);
	md_gaussian_rand(N, dims, iptr1_cpu);

	md_copy(N, dims, optr_gpu, optr_cpu, CFL_SIZE);
	md_copy(N, dims, iptr1_gpu, iptr1_cpu, CFL_SIZE);

	function(N, dims, strs, optr_gpu, strs, iptr1_gpu);
	function(N, dims, strs, optr_cpu, strs, iptr1_cpu);

	md_copy(N, dims, optr_gpu_cpu, optr_gpu, CFL_SIZE);

	float err = md_znrmse(N, dims, optr_cpu, optr_gpu_cpu);

	md_free(optr_cpu);
	md_free(iptr1_cpu);

	md_free(optr_gpu_cpu);

	md_free(optr_gpu);
	md_free(iptr1_gpu);

	if ((UT_TOL < err) || (!safe_isfinite(err)))
		debug_printf(DP_WARN, "err: %e\n", err);

	return (UT_TOL >= err) && safe_isfinite(err);
}

static bool test_md_zsqrt2(void) { UT_ASSERT(test_md_z2op(md_zsqrt2));}
UT_GPU_REGISTER_TEST(test_md_zsqrt2);

static bool test_md_zabs2(void) { UT_ASSERT(test_md_z2op(md_zabs2));}
UT_GPU_REGISTER_TEST(test_md_zabs2);

static bool test_md_zconj2(void) { UT_ASSERT(test_md_z2op(md_zconj2));}
UT_GPU_REGISTER_TEST(test_md_zconj2);

static bool test_md_zreal2(void) { UT_ASSERT(test_md_z2op(md_zreal2));}
UT_GPU_REGISTER_TEST(test_md_zreal2);

static bool test_md_zimag2(void) { UT_ASSERT(test_md_z2op(md_zimag2));}
UT_GPU_REGISTER_TEST(test_md_zimag2);

static bool test_md_zexpj2(void) { UT_ASSERT(test_md_z2op(md_zexpj2));}
UT_GPU_REGISTER_TEST(test_md_zexpj2);

static bool test_md_zexp2(void) { UT_ASSERT(test_md_z2op(md_zexp2));}
UT_GPU_REGISTER_TEST(test_md_zexp2);

static bool test_md_zlog2(void) { UT_ASSERT(test_md_z2op(md_zlog2));}
UT_GPU_REGISTER_TEST(test_md_zlog2);

static bool test_md_zarg2(void) { UT_ASSERT(test_md_z2op(md_zarg2));}
UT_GPU_REGISTER_TEST(test_md_zarg2);

static bool test_md_zsin2(void) { UT_ASSERT(test_md_z2op(md_zsin2));}
UT_GPU_REGISTER_TEST(test_md_zsin2);

static bool test_md_zcos2(void) { UT_ASSERT(test_md_z2op(md_zcos2));}
UT_GPU_REGISTER_TEST(test_md_zcos2);

static bool test_md_zsinh2(void) { UT_ASSERT(test_md_z2op(md_zsinh2));}
UT_GPU_REGISTER_TEST(test_md_zsinh2);

static bool test_md_zcosh2(void) { UT_ASSERT(test_md_z2op(md_zcosh2));}
UT_GPU_REGISTER_TEST(test_md_zcosh2);







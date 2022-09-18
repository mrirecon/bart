
#include "num/flpmath.h"
#include "num/multind.h"
#include "num/convcorr.h"
#include "num/init.h"

#include "utest.h"

#ifdef USE_CUDNN
int cudnn = 1;
#else
int cudnn = 0;
#endif

static bool test_convcorr_cf_1D(void)
{
	num_init_gpu();

	enum {N = 6};
	long odims[N] = {2, 1, 3, 1, 1, 4};
	long idims[N] = {1, 5, 5, 1, 1, 4};
	long kdims[N] = {2, 5, 3, 1, 1, 1};

	bool test = true;

	test = test && test_zconvcorr_fwd(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	test = test && test_zconvcorr_bwd_in(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	test = test && test_zconvcorr_bwd_krn(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	UT_ASSERT(test);
}
UT_GPU_REGISTER_TEST(test_convcorr_cf_1D);


static bool test_convcorr_cf_2D(void)
{
	num_init_gpu();

	enum {N = 6};
	long odims[N] = {2, 1, 3, 2, 1, 4};
	long idims[N] = {1, 5, 5, 5, 1, 4};
	long kdims[N] = {2, 5, 3, 4, 1, 1};

	bool test = true;

	test = test && test_zconvcorr_fwd(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	test = test && test_zconvcorr_bwd_in(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	test = test && test_zconvcorr_bwd_krn(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	UT_ASSERT(test);
}
UT_GPU_REGISTER_TEST(test_convcorr_cf_2D);


static bool test_convcorr_cf_3D(void)
{
	num_init_gpu();

	enum {N = 6};
	long odims[N] = {2, 1, 3, 2, 1, 4};
	long idims[N] = {1, 5, 5, 5, 4, 4};
	long kdims[N] = {2, 5, 3, 4, 4, 1};

	bool test = true;

	test = test && test_zconvcorr_fwd(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	test = test && test_zconvcorr_bwd_in(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	test = test && test_zconvcorr_bwd_krn(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	UT_ASSERT(test);
}
UT_GPU_REGISTER_TEST(test_convcorr_cf_3D);


static bool test_convcorr_rand_ord(void)
{
	num_init_gpu();

	enum {N = 6};
	long odims[N] = {2, 4, 3, 1, 2, 1};
	long idims[N] = {1, 4, 5, 5, 5, 4};
	long kdims[N] = {2, 1, 3, 5, 4, 4};

	bool test = true;

	test = test && test_zconvcorr_fwd(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 0 + cudnn);

	test = test && test_zconvcorr_bwd_in(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 0 + cudnn);

	test = test && test_zconvcorr_bwd_krn(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 0 + cudnn);

	UT_ASSERT(test);
}

UT_GPU_REGISTER_TEST(test_convcorr_rand_ord);


static bool test_convcorr_cf_one_channel(void)
{
	num_init_gpu();

	enum {N = 6};
	long odims[N] = {2, 1, 3, 2, 1, 4};
	long idims[N] = {1, 1, 5, 5, 4, 4};
	long kdims[N] = {2, 1, 3, 4, 4, 1};

	bool test = true;

	test = test && test_zconvcorr_fwd(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	test = test && test_zconvcorr_bwd_in(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1);

	test = test && test_zconvcorr_bwd_krn(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, NULL, NULL, false,
						1.e-6, true, 1 + cudnn);

	UT_ASSERT(test);
}

UT_GPU_REGISTER_TEST(test_convcorr_cf_one_channel);



static bool test_convcorr_cf_dil_strs(void)
{
	num_init_gpu();

	enum {N = 6};
	long odims[N] = 	{2, 1, 3, 3, 1, 4};
	long idims[N] = 	{1, 5, 5, 7, 4, 4};
	long kdims[N] = 	{2, 5, 2, 3, 4, 1};

	long dilation[N] =	{1, 1, 2, 1, 1, 1};
	long strides[N] =	{1, 1, 1, 2, 1, 1};


	bool test = true;

	test = test && test_zconvcorr_fwd(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, dilation, strides, false,
						1.e-6, true, 1 + cudnn);

#ifdef NON_DETERMINISTIC
	int no_bwd_algo = 1 + cudnn;
#else
	int no_bwd_algo = 0;
#endif

	test = test && test_zconvcorr_bwd_in(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, dilation, strides, false,
						1.e-6, true, no_bwd_algo);

	test = test && test_zconvcorr_bwd_krn(	N,
						odims, MD_STRIDES(N, odims, CFL_SIZE),
						idims, MD_STRIDES(N, idims, CFL_SIZE),
						kdims, MD_STRIDES(N, kdims, CFL_SIZE),
						28, dilation, strides, false,
						1.e-6, true, 1 + cudnn);

	UT_ASSERT(test);
}
UT_GPU_REGISTER_TEST(test_convcorr_cf_dil_strs);


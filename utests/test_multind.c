
#include <complex.h>

#include "num/multind.h"
#include "num/rand.h"
#include "num/flpmath.h"


#include "utest.h"


static bool test_md_copy(void)
{
	enum { N = 4 };
	long dims[N] = { 10, 10, 10, 10 };

	complex float* a = md_alloc(N, dims, sizeof(complex float));

	md_gaussian_rand(N, dims, a);

	complex float* b = md_alloc(N, dims, sizeof(complex float));

	md_copy(N, dims, b, a, sizeof(complex float));

	bool eq = md_compare(N, dims, a, b, sizeof(complex float));

	md_free(a);
	md_free(b);

	return eq;
}


UT_REGISTER_TEST(test_md_copy);


static bool test_md_transpose(void)
{
	enum { N = 4 };
	long dims[N] = { 10, 10, 10, 10 };

	complex float* a = md_alloc(N, dims, sizeof(complex float));

	md_gaussian_rand(N, dims, a);

	complex float* b = md_alloc(N, dims, sizeof(complex float));
	complex float* c = md_alloc(N, dims, sizeof(complex float));

	md_transpose(N, 0, 2, dims, b, dims, a, sizeof(complex float));
	md_transpose(N, 0, 2, dims, c, dims, b, sizeof(complex float));

	bool eq = md_compare(N, dims, a, c, sizeof(complex float));

	md_free(a);
	md_free(b);
	md_free(c);

	return eq;
}


UT_REGISTER_TEST(test_md_transpose);


static bool test_md_swap(void)
{
	enum { N = 4 };
	long dims[N] = { 10, 10, 10, 10 };

	complex float* a = md_alloc(N, dims, sizeof(complex float));
	complex float* b = md_alloc(N, dims, sizeof(complex float));
	complex float* c = md_alloc(N, dims, sizeof(complex float));

	md_gaussian_rand(N, dims, a);
	md_gaussian_rand(N, dims, b);
	md_gaussian_rand(N, dims, c);

	complex float* d = md_alloc(N, dims, sizeof(complex float));
	complex float* e = md_alloc(N, dims, sizeof(complex float));
	complex float* f = md_alloc(N, dims, sizeof(complex float));

	md_copy(N, dims, d, a, sizeof(complex float));
	md_copy(N, dims, e, b, sizeof(complex float));
	md_copy(N, dims, f, c, sizeof(complex float));

	md_circular_swap(3, N, dims, (void*[]){ a, b, c }, sizeof(complex float));

	bool eq = true;
	eq &= md_compare(N, dims, c, d, sizeof(complex float));
	eq &= md_compare(N, dims, a, e, sizeof(complex float));
	eq &= md_compare(N, dims, b, f, sizeof(complex float));

	md_free(a);
	md_free(b);
	md_free(c);
	md_free(d);
	md_free(e);
	md_free(f);

	return eq;
}


UT_REGISTER_TEST(test_md_swap);



static bool test_md_flip(void)
{
	enum { N = 4 };
	long dims[N] = { 10, 10, 10, 10 };

	complex float* a = md_alloc(N, dims, sizeof(complex float));

	md_gaussian_rand(N, dims, a);

	complex float* b = md_alloc(N, dims, sizeof(complex float));

	md_flip(N, dims, MD_BIT(0) | MD_BIT(2), b, a, sizeof(complex float));
	md_flip(N, dims, MD_BIT(0) | MD_BIT(2), b, b, sizeof(complex float));

	bool eq = md_compare(N, dims, a, b, sizeof(complex float));

	md_free(a);
	md_free(b);

	return eq;
}


UT_REGISTER_TEST(test_md_flip);



static bool test_md_reshape(void)
{
	enum { N = 4 };
	long dims1[N] = { 10, 10, 10, 10 };
	long dims2[N] = { 10, 20, 10,  5 };
	long dims3[N] = {  5, 20, 20,  5 };
	long dims4[N] = {  5, 10, 20, 10 };

	complex float* a = md_alloc(N, dims1, sizeof(complex float));
	complex float* b = md_alloc(N, dims1, sizeof(complex float));
	complex float* c = md_alloc(N, dims1, sizeof(complex float));

	md_gaussian_rand(N, dims1, a);

	md_reshape(N, MD_BIT(1)|MD_BIT(3), dims2, b, dims1, a, sizeof(complex float));
	md_reshape(N, MD_BIT(0)|MD_BIT(2), dims3, c, dims2, b, sizeof(complex float));
	md_reshape(N, MD_BIT(1)|MD_BIT(3), dims4, b, dims3, c, sizeof(complex float));
	md_reshape(N, MD_BIT(0)|MD_BIT(2), dims1, c, dims4, b, sizeof(complex float));

	bool eq = md_compare(N, dims1, a, c, sizeof(complex float));

	md_free(a);
	md_free(b);
	md_free(c);

	return eq;
}


UT_REGISTER_TEST(test_md_reshape);


static bool test_compress(void)
{
	enum { N = 1 };
	long dims[N] = { 31 };

	complex float* _ptr1 = md_alloc(N, dims, CFL_SIZE);
	md_gaussian_rand(N, dims, _ptr1);

	dims[0] = 62;

	float* ptr1 = (float*)_ptr1;
	md_sgreatequal(N, dims, ptr1, ptr1, 0.);

	long M = (dims[0] + 31) / 32;
	uint32_t (*compress)[M] = xmalloc(sizeof *compress);

	md_mask_compress(N, dims, M, *compress, ptr1);

	float* ptr2 = md_alloc(N, dims, FL_SIZE);
	md_mask_decompress(N, dims, ptr2, M, *compress);

	float err = md_nrmse(N, dims, ptr2, ptr1);

	md_free(ptr1);
	md_free(ptr2);
	xfree(compress);
	
	UT_RETURN_ASSERT(err < UT_TOL);
}

UT_REGISTER_TEST(test_compress);


static float test_md_reflectpad_center(int D,
		const long odims[D], const complex float* expect,
		const long idims[D], const complex float* in)
{
	complex float* t = md_alloc(D, odims, sizeof(complex float));

	md_reflectpad_center(D, odims, t, idims, in, sizeof(complex float));

	float err = md_znrmse(D, odims, expect, t);

	md_free(t);

	return err;
}

static bool test_md_reflectpad_center_1(void)
{
	const long idims[] = { 3, 2 };

	const complex float in[] = {
		1, 2, 3,
		4, 5, 6,
	};

	const long odims[] = { 4, 3 };

	const complex float good[] = {
		1, 1, 2, 3,
		4, 4, 5, 6,
		4, 4, 5, 6,
	};

	UT_RETURN_ASSERT(UT_TOL > test_md_reflectpad_center(2, odims, good, idims, in));
}

UT_REGISTER_TEST(test_md_reflectpad_center_1);


static bool test_md_reflectpad_center_2(void)
{
	const long idims[] = { 2, 2 };

	complex float in[] = {
		1, 2,
		4, 5,
	};

	const long odims[] = { 9, 4 };

	complex float good[] = {
		2, 2, 1, 1, 2, 2, 1, 1, 2,
		2, 2, 1, 1, 2, 2, 1, 1, 2,
		5, 5, 4, 4, 5, 5, 4, 4, 5,
		5, 5, 4, 4, 5, 5, 4, 4, 5,
	};

	UT_RETURN_ASSERT(UT_TOL > test_md_reflectpad_center(2, odims, good, idims, in));
}

UT_REGISTER_TEST(test_md_reflectpad_center_2);


static bool test_md_next(void)
{
	const long dims[] = { 2, 3 };

	const long good[6][2] = {
		{ 0, 0 }, { 1, 0 },
		{ 0, 1 }, { 1, 1 },
		{ 0, 2 }, { 1, 2 },
	};

	long pos[2] = { 0 };

	int i = 0;

	do {
		UT_RETURN_ON_FAILURE(md_check_equal_dims(2, good[i++], pos, 3UL));

	} while (md_next(2, dims, 3UL, pos));

	return true;
}

UT_REGISTER_TEST(test_md_next);


static bool test_md_next_permuted_1(void)
{
	const long dims[] = { 2, 4, 3 };

	const int order[] = { 2, 0, 1 };

	const long good[6][3] = {
		{ 0, 0, 0 }, { 0, 0, 1 }, { 0, 0, 2 },
		{ 1, 0, 0 }, { 1, 0, 1 }, { 1, 0, 2 }
	};

	long pos[3] = { 0 };

	int i = 0;

	do {
		UT_RETURN_ON_FAILURE(md_check_equal_dims(3, good[i++], pos, 7UL));

	} while (md_next_permuted(3, order, dims, 5UL, pos));

	return true;
}

UT_REGISTER_TEST(test_md_next_permuted_1);

static bool test_md_next_permuted_2(void)
{
	const long dims[] = { 2, 4, 3 };

	const int order[] = { 2, 0, 1 };

	const long good[8][3] = {
		{ 0, 0, 0 }, { 1, 0, 0 }, 
		{ 0, 1, 0 }, { 1, 1, 0 },
		{ 0, 2, 0 }, { 1, 2, 0 },
		{ 0, 3, 0 }, { 1, 3, 0 }
	};

	long pos[3] = { 0 };

	int i = 0;

	do {
		UT_RETURN_ON_FAILURE(md_check_equal_dims(3, good[i++], pos, 7UL));

	} while (md_next_permuted(3, order, dims, 3UL, pos));

	return true;
}

UT_REGISTER_TEST(test_md_next_permuted_2);

static bool test_md_permute_dims_inverse(void)
{
	const int order[6] = { 0, 2, 5, 3, 1, 4 };
	const long good[6] = { 0, 1, 2, 3, 4, 5 };

	long permute[6];
	long inv_permute[6];

	md_permute_dims(6, order, permute, good);

	int inv_order[6];
	md_permute_invert(6, inv_order, order);
	md_permute_dims(6, inv_order, inv_permute, permute);

	return md_check_equal_dims(6, good, inv_permute, ~0UL);
}

UT_REGISTER_TEST(test_md_permute_dims_inverse);


static bool test_md_permute_flags(void)
{
	const int order[6] = { 0, 2, 5, 3, 1, 4 };
	unsigned long in = MD_BIT(1) | MD_BIT(5);
	unsigned long out = MD_BIT(2) | MD_BIT(4);

	return out == md_permute_flags(6, order, in);
}

UT_REGISTER_TEST(test_md_permute_flags);


static bool test_md_unravel_index_permuted(void)
{
	const long dims[] = { 2, 4, 3 };

	const int order[] = { 2, 0, 1 };

	const long good[3] = { 0, 1, 2 };

	long pos[3] = { 0 };

	long idx = 8;

	md_unravel_index_permuted(3, pos, 7UL, dims, idx, order);
	
	UT_RETURN_ON_FAILURE(md_check_equal_dims(3, good, pos, 7UL));

	return true;
}

UT_REGISTER_TEST(test_md_unravel_index_permuted);


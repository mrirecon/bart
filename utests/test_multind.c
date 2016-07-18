
#include <complex.h>

#include "num/multind.h"
#include "num/rand.h"

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



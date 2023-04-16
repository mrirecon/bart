

#include "num/rand.h"

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/tree.h"

#include "utest.h"

static int cmp(const void* _a, const void* _b)
{
	float a = *(float*)_a;
	float b = *(float*)_b;

	if (a > b)
		return 1;
	
	if (a < b)
		return -1;
	
	return 0;
}

static int cmp_range(const void* _a, const void* _b)
{
	float a = ((float*)_a)[0];

	float min = ((float*)_b)[0];
	float max = ((float*)_b)[1];

	if (a > max)
		return 1;

	if (a < min)
		return -1;

	return 0;
}


static bool test_tree_sorted(void)
{
	int N = 100;
	float vals[N];

	tree_t tree = tree_create(cmp);
	
	for (int i = 0; i < (int)ARRAY_SIZE(vals); i++) {

		vals[i] = gaussian_rand();
		tree_insert(tree, vals + i);
	}

	float* vals_sorted[N];
	long NR = tree_count(tree);

	NR = tree_count(tree);
	tree_to_array(tree, NR, (void**)vals_sorted);

	for (int i = 1; i < NR; i++)
		if (*(vals_sorted[i]) < *(vals_sorted[i-1]))
			return false;

	float b1[] = { -.5, -.1 };
	while (NULL != tree_find_min(tree, b1, cmp_range, true));

	NR = tree_count(tree);
	tree_to_array(tree, NR, (void**)vals_sorted);

	for (int i = 1; i < NR; i++)
		if (*(vals_sorted[i]) < *(vals_sorted[i-1]))
			return false;

	float b2[] = { .1, .5 };
	while (NULL != tree_find(tree, b2, cmp_range, true));

	NR = tree_count(tree);
	tree_to_array(tree, NR, (void**)vals_sorted);

	//for (int i = 0; i < NR; i++)
	//	debug_printf(DP_INFO, "%f\n", *(vals_sorted[i]));

	for (int i = 1; i < NR; i++)
		if (*(vals_sorted[i]) < *(vals_sorted[i-1]))
			return false;

	tree_free(tree);

	return true;
}


UT_REGISTER_TEST(test_tree_sorted);




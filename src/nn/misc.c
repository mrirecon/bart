
#include <math.h>
#include <complex.h>

#include "misc/misc.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc.h"


void onehotenc_to_index(unsigned int N, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	int class_index = -1;
	
	for (unsigned int i = 0; i < N; i++) {

		if (odims[i] != idims[i]) {

			assert(-1 == class_index);
			class_index = i;
		}
	}

	assert(-1 != class_index);

	long num_classes = idims[class_index];

	long ostrs[N];
	long istrs[N];
	long pos[N];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, idims, CFL_SIZE);
	md_singleton_strides(N, pos);

	do {
		long tpos [N];
		md_copy_dims(N, tpos, pos);

		long index = 0;
		bool double_max = false;
		float max_val = crealf(MD_ACCESS(N, istrs, tpos, src));

		tpos[class_index]++;

		for (; tpos[class_index] < num_classes; tpos[class_index]++) {

			if (crealf(MD_ACCESS(N, istrs, tpos, src)) == max_val)
				double_max = true;

			if (crealf(MD_ACCESS(N, istrs, tpos, src)) > max_val) {

				index = tpos[class_index];
				max_val = crealf(MD_ACCESS(N, istrs, tpos, src));
				double_max = false;
			}
		}

		if (double_max) {

			debug_printf(DP_WARN, "One-Hot Encoding to Index has found multiple maximal values. Took the first index:\n");
			debug_print_dims(DP_INFO, N, tpos);
		}

		MD_ACCESS(N, ostrs, pos, dst) = index;
			
	} while (md_next(N, odims, ~0, pos));
}

void index_to_onehotenc(unsigned int N, const long odims[N], complex float* dst, const long idims[N], const complex float* src)
{
	int class_index = -1;
	
	for (unsigned int i = 0; i < N; i++) {

		if (odims[i] != idims[i]) {

			assert(-1 == class_index);
			class_index = i;
		}
	}

	assert(-1 != class_index);

	long num_classes = odims[class_index];

	long ostrs[N];
	long istrs[N];
	long pos[N];

	md_calc_strides(N, ostrs, odims, CFL_SIZE);
	md_calc_strides(N, istrs, idims, CFL_SIZE);
	md_singleton_strides(N, pos);

	md_clear(N, odims, dst, CFL_SIZE);

	do {
		long tpos [N];
		md_copy_dims(N, tpos, pos);
		tpos[class_index] = lroundf(MD_ACCESS(N, istrs, tpos, src));

		assert(tpos[class_index] < num_classes);
		assert(0 <= tpos[class_index]);

		MD_ACCESS(N, ostrs, tpos, dst) = 1.;
	
	} while (md_next(N, idims, ~0, pos));
}


void onehotenc_set_max_to_one(unsigned int N, const long dims[N], unsigned int class_index, complex float* dst, const complex float* src)
{
	long bdims[N];
	md_select_dims(N, ~MD_BIT(class_index), bdims, dims);

	long strs[N];
	long bstrs[N];
	md_calc_strides(N, strs, dims, CFL_SIZE);
	md_calc_strides(N, bstrs, bdims, CFL_SIZE);

	complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);
	md_zreal(N, dims, tmp, src);

	complex float* max = md_alloc_sameplace(N, bdims, CFL_SIZE, dst);
	md_copy2(N, bdims, bstrs, max, strs, tmp, CFL_SIZE);

	md_zmax2(N, dims, bstrs, max, bstrs, max, strs, tmp);

	md_zgreatequal2(N, dims, strs, dst, strs, tmp, bstrs, max);

	md_free(tmp);
	md_free(max);
}


float onehotenc_accuracy(unsigned int N, const long dims[N], unsigned int class_index, const complex float* cmp, const complex float* ref)
{
	complex float* tmp_cmp = md_alloc_sameplace(N, dims, CFL_SIZE, cmp);
	complex float* tmp_ref = md_alloc_sameplace(N, dims, CFL_SIZE, ref);

	onehotenc_set_max_to_one(N, dims, class_index, tmp_cmp, cmp);
	onehotenc_set_max_to_one(N, dims, class_index, tmp_ref, ref);

	float result = powf(crealf(md_zrmse(N, dims, tmp_cmp, tmp_ref)), 2.) /2. * dims[class_index];

	md_free(tmp_cmp);
	md_free(tmp_ref);

	return 1. - result;
}
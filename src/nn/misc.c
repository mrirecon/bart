/* Copyright 2021-2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <math.h>
#include <complex.h>
#include <stdio.h>

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

void onehotenc_confusion_matrix(unsigned int N, const long dims[N], unsigned int class_index, complex float* dst, const complex float* pred, const complex float* ref)
{
	long classes = dims[class_index];

	complex float* tmp_pred = md_alloc_sameplace(N, dims, CFL_SIZE, pred);
	complex float* tmp_ref = md_alloc_sameplace(N, dims, CFL_SIZE, ref);

	onehotenc_set_max_to_one(N, dims, class_index, tmp_pred, pred);
	onehotenc_set_max_to_one(N, dims, class_index, tmp_ref, ref);

	long tdims[N + 2];
	long ostrs[N + 2];
	long pstrs[N + 2];
	long rstrs[N + 2];

	md_singleton_strides(N + 2, ostrs);
	md_singleton_strides(N + 2, pstrs);
	md_singleton_strides(N + 2, rstrs);

	md_copy_dims(N, tdims + 2, dims);
	tdims[0] = classes;
	tdims[1] = classes;

	md_calc_strides(2, ostrs, tdims, CFL_SIZE);
	md_calc_strides(N, rstrs + 2, dims, CFL_SIZE);
	md_calc_strides(N, pstrs + 2, dims, CFL_SIZE);

	tdims[class_index + 2] = 1;
	pstrs[0] = pstrs[class_index + 2];
	rstrs[1] = pstrs[class_index + 2];

	md_ztenmul2(2 + N, tdims, ostrs, dst, pstrs, tmp_pred, rstrs, tmp_ref);

	md_free(tmp_pred);
	md_free(tmp_ref);
}


extern void print_confusion_matrix(unsigned int N, const long dims[N], unsigned int class_index, const complex float* pred, const complex float* ref)
{
	long classes = dims[class_index];

	complex float matrix[classes][classes];
	onehotenc_confusion_matrix(N, dims, class_index, &(matrix[0][0]), pred, ref);

	complex float* tmp_cmp = md_alloc_sameplace(N, dims, CFL_SIZE, pred);
	complex float* tmp_ref = md_alloc_sameplace(N, dims, CFL_SIZE, ref);

	onehotenc_set_max_to_one(N, dims, class_index, tmp_cmp, pred);
	onehotenc_set_max_to_one(N, dims, class_index, tmp_ref, ref);

	complex float pred_count[classes];
	complex float ref_count[classes];

	md_zsum(N, dims, ~MD_BIT(class_index), pred_count, tmp_cmp);
	md_zsum(N, dims, ~MD_BIT(class_index), ref_count, tmp_ref);

	md_free(tmp_cmp);
	md_free(tmp_ref);

	long N_pred = md_calc_size(N, dims) / classes;
	int count_char = MAX(3, snprintf(NULL, 0, "%ld", N_pred));

	printf("\npred \\ ref |");
	for (int i = 0; i < classes; i++)
		printf("%*d", count_char + 1, i);
	printf("|%*s\n", (count_char + 1), "sum");

	for (int i = 0; i < 11; i++)
		printf("%c", '-');
	printf("|");
	for (int i = 0; i < (int)classes * (count_char + 1); i++)
		printf("%c", '-');
	printf("|");
	for (int i = 0; i < (count_char + 1); i++)
		printf("%c", '-');
	printf("\n");

	for (int i = 0; i < classes; i++) {

		printf("%-11d|", i);

		for (int j = 0; j < classes; j++)
			printf("%*ld", count_char + 1, (long)crealf(matrix[j][i]));

		printf("|%*ld\n", count_char + 1, (long)crealf(pred_count[i]));
	}

	for (int i = 0; i < 11; i++)
		printf("%c", '-');
	printf("|");
	for (int i = 0; i < (int)classes * (count_char + 1); i++)
		printf("%c", '-');
	printf("|");
	for (int i = 0; i < (count_char + 1); i++)
		printf("%c", '-');
	printf("\n");

	printf("%-11s|", "sum");
	for (int i = 0; i < classes; i++)
		printf("%*ld", count_char + 1, (long)crealf(ref_count[i]));
	printf("|%*ld\n", count_char + 1, N_pred);
}

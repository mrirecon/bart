

#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#include "num/gpukrnls_triagmat.h"
#endif

#include "triagmat.h"


// lapack convention packed storage converted to 0 based indexing
// (netlib.org/lapack/lug/node123.html)
long upper_triag_idx(long i, long j)
{
	if (i > j)
		return -1 * upper_triag_idx(j, i);

	return i + ((j + 1) * j) / 2;
}

// Integer square root
// (using linear search, descending)
static long isqrt(long y)
{
	assert(0 <= y);

	long ret = lround(sqrt(y));
	assert(y == ret * ret);

	return ret;
}


complex float* hermite_to_uppertriag(int dim1, int dim2, int dimt, int N, long out_dims[N], const long* dims, const complex float* src)
{
	assert(dims[dim1] == dims[dim2]);

	long dim = (dim1 == dim2) ? isqrt(dims[dim1]) : dims[dim1];

	md_copy_dims(N, out_dims, dims);
	out_dims[dim1] = 1;
	out_dims[dim2] = 1;

	assert(1 == out_dims[dimt]);

	out_dims[dimt] = (dim * (dim + 1)) / 2;

	complex float* out = md_alloc_sameplace(N, out_dims, CFL_SIZE, src);

	long slc_dims[N];
	md_select_dims(N, ~MD_BIT(dimt), slc_dims, out_dims);

	long ipos[N];
	long opos[N];

	md_set_dims(N, ipos, 0);
	md_set_dims(N, opos, 0);

	for (int i = 0; i < dim; i++) {

		for (int j = 0; j < dim; j++) {

			if (i > j)
				continue;

			opos[dim1] = upper_triag_idx(i, j);


			if (dim1 == dim2) {

				ipos[dim1] = i + dim * j;
			} else {

				ipos[dim1] = i;
				ipos[dim2] = j;
			}

			md_move_block(N, slc_dims, opos, out_dims, out, ipos, dims, src, CFL_SIZE);
		}
	}

	return out;
}

complex float* uppertriag_to_hermite(int dim1, int dim2, int dimt, int N, long out_dims[N], const long* dims, const complex float* src)
{
	long ldim = 1;
	long udim = dims[dimt];

	long dim = 1;

	while (ldim < udim) {

		dim = (ldim + udim) / 2;

		if ((dim * (dim + 1)) / 2 == dims[dimt])
			break;

		if ((dim * (dim + 1)) / 2 < dims[dimt])
			ldim = dim;
		else
			udim = dim;
	}

	md_copy_dims(N, out_dims, dims);
	out_dims[dimt] = 1;

	assert(1 == out_dims[dim1]);
	assert(1 == out_dims[dim2]);

	out_dims[dim1] = dim;
	out_dims[dim2] *= dim;

	complex float* out = md_alloc_sameplace(N, out_dims, CFL_SIZE, src);

	long slc_dims[N];
	md_select_dims(N, ~MD_BIT(dimt), slc_dims, dims);

	long ipos[N];
	long opos[N];

	md_set_dims(N, ipos, 0);
	md_set_dims(N, opos, 0);

	long ostrs[N];
	md_calc_strides(N, ostrs, out_dims, CFL_SIZE);

	for (int i = 0; i < dim; i++) {

		for (int j = 0; j < dim; j++) {

			ipos[dimt] = labs(upper_triag_idx(i, j));

			if (dim1 == dim2) {

				opos[dim1] = i + dim * j;
			} else {

				opos[dim1] = i;
				opos[dim2] = j;
			}

			md_move_block(N, slc_dims, opos, out_dims, out, ipos, dims, src, CFL_SIZE);

			if (0 > upper_triag_idx(i, j)) {

				md_zconj2(N, slc_dims, ostrs, &MD_ACCESS(N, ostrs, opos, out), ostrs, &MD_ACCESS(N, ostrs, opos, out));
			}
		}
	}

	return out;
}

float* symmetric_to_uppertriag(int dim1, int dim2, int dimt, int N, long out_dims[N], const long* dims, const float* src)
{
	assert(dims[dim1] == dims[dim2]);

	long dim = (dim1 == dim2) ? isqrt(dims[dim1]) : dims[dim1];

	md_copy_dims(N, out_dims, dims);
	out_dims[dim1] = 1;
	out_dims[dim2] = 1;

	assert(1 == out_dims[dimt]);

	out_dims[dimt] = (dim * (dim + 1)) / 2;

	float* out = md_alloc_sameplace(N, out_dims, FL_SIZE, src);

	long slc_dims[N];
	md_select_dims(N, ~MD_BIT(dimt), slc_dims, out_dims);

	long ipos[N];
	long opos[N];

	md_set_dims(N, ipos, 0);
	md_set_dims(N, opos, 0);

	for (int i = 0; i < dim; i++) {

		for (int j = 0; j < dim; j++) {

			if (i > j)
				continue;

			opos[dim1] = upper_triag_idx(i, j);


			if (dim1 == dim2) {

				ipos[dim1] = i + dim * j;
			} else {

				ipos[dim1] = i;
				ipos[dim2] = j;
			}

			md_move_block(N, slc_dims, opos, out_dims, out, ipos, dims, src, FL_SIZE);
		}
	}

	return out;
}

float* uppertriag_to_symmetric(int dim1, int dim2, int dimt, int N, long out_dims[N], const long* dims, const float* src)
{
	long ldim = 1;
	long udim = dims[dimt];

	long dim = 1;

	while (ldim < udim) {

		dim = (ldim + udim) / 2;

		if ((dim * (dim + 1)) / 2 == dims[dimt])
			break;

		if ((dim * (dim + 1)) / 2 < dims[dimt])
			ldim = dim;
		else
			udim = dim;
	}

	md_copy_dims(N, out_dims, dims);
	out_dims[dimt] = 1;

	assert(1 == out_dims[dim1]);
	assert(1 == out_dims[dim2]);

	out_dims[dim1] = dim;
	out_dims[dim2] *= dim;

	float* out = md_alloc_sameplace(N, out_dims, FL_SIZE, src);

	long slc_dims[N];
	md_select_dims(N, ~MD_BIT(dimt), slc_dims, dims);

	long ipos[N];
	long opos[N];

	md_set_dims(N, ipos, 0);
	md_set_dims(N, opos, 0);

	long ostrs[N];
	md_calc_strides(N, ostrs, out_dims, FL_SIZE);

	for (int i = 0; i < dim; i++) {

		for (int j = 0; j < dim; j++) {

			ipos[dimt] = labs(upper_triag_idx(i, j));

			if (dim1 == dim2) {

				opos[dim1] = i + dim * j;
			} else {

				opos[dim1] = i;
				opos[dim2] = j;
			}

			md_move_block(N, slc_dims, opos, out_dims, out, ipos, dims, src, FL_SIZE);
		}
	}

	return out;
}


void md_ztenmul_upper_triag2(int dim1, int dim2, int N, const long dims[N], const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src, const long /*mdims*/[N], const long mstrs[N], const complex float* mat)
{
	md_clear2(N, dims, ostrs, dst, CFL_SIZE);

	long slc_dims[N];
	md_select_dims(N, ~(MD_BIT(dim1) | MD_BIT(dim2)), slc_dims, dims);

	long pos[N];
	md_set_dims(N, pos, 0);

	do {

		long offset = labs(upper_triag_idx(pos[dim1], pos[dim2])) * MAX(mstrs[dim1], mstrs[dim2]) / (long)CFL_SIZE;

		(0 > upper_triag_idx(pos[dim1], pos[dim2]) ? md_zfmac2 : md_zfmacc2)(N, slc_dims, ostrs, &MD_ACCESS(N, ostrs, pos, dst), istrs, &MD_ACCESS(N, istrs, pos, src), mstrs, mat + offset);

	} while (md_next(N, dims, MD_BIT(dim1) | MD_BIT(dim2), pos));
}

void md_ztenmul_upper_triag(int dim1, int dim2, int N, const long odims[N], complex float* dst, const long idims[N], const complex float* src, const long mdims[N], const complex float* mat)
{
	long msize = MAX(MAX(MAX(odims[dim1], odims[dim2]), idims[dim1]), idims[dim2]);

	assert(1 == odims[dim1] || msize == odims[dim1]);
	assert(1 == odims[dim2] || msize == odims[dim2]);
	assert(1 == idims[dim1] || msize == idims[dim1]);
	assert(1 == idims[dim2] || msize == idims[dim2]);
	assert(mdims[dim1] * mdims[dim2] == msize * (msize + 1) / 2);
	assert(1 == mdims[dim1] || 1 == mdims[dim2]);

	long max_dims[N];
	md_select_dims(N, ~(MD_BIT(dim1) | MD_BIT(dim2)), max_dims, mdims);

	assert(md_check_compat(N, ~0UL, odims, idims));
	assert(md_check_compat(N, ~0UL, max_dims, odims));
	assert(md_check_compat(N, ~0UL, max_dims, idims));

	md_max_dims(N, ~0UL, max_dims, max_dims, odims);
	md_max_dims(N, ~0UL, max_dims, max_dims, idims);

	long ostrs[N];
	md_calc_strides(N, ostrs, odims, CFL_SIZE);

	long istrs[N];
	md_calc_strides(N, istrs, idims, CFL_SIZE);

	long mstrs[N];
	md_calc_strides(N, mstrs, mdims, CFL_SIZE);

	md_ztenmul_upper_triag2(dim1, dim2, N, max_dims, ostrs, dst, istrs, src, mdims, mstrs, mat);
}


static void vptr_md_fmac_upper_triag2(int dim1, int dim2, int N, int D, const long* dims[N], const long* strs[N], void* args[N])
{

#ifdef USE_CUDA
	if (   (D > 5) && (5 == dim1) && (6 == dim2)
	    && (2 == dims[0][D - 1]) && ((long)FL_SIZE == strs[0][D - 1]) && ((long)FL_SIZE == strs[1][D - 1]) && (0 == strs[2][D - 1])
	    && (1 == dims[0][4])
	    && (4 <= md_calc_blockdim(D, dims[0], strs[0], CFL_SIZE))
	    && (4 <= md_calc_blockdim(D, dims[1], strs[1], CFL_SIZE))
	    && (4 <= md_calc_blockdim(D, dims[2], strs[2],  FL_SIZE))
	    && (0 == (md_nontriv_strides(2, strs[0] + 5) & md_nontriv_strides(2, strs[1] + 5)))
	    && (3 == (md_nontriv_strides(2, strs[0] + 5) | md_nontriv_strides(2, strs[1] + 5)))
	    && (md_nontriv_strides(2, strs[0] + 5) != md_nontriv_strides(2, strs[1] + 5))
	    && cuda_ondevice(args[0])) {

		long pos[D];
		md_set_dims(D, pos, 0);

		do {
			cuda_zrfmac_upper_triagmat(md_calc_size(3, dims[0]), dims[0][3], MAX(dims[0][5], dims[0][6]),
							MAX(strs[0][5], strs[0][6]) / (long)FL_SIZE,
							MAX(strs[1][5], strs[1][6]) / (long)FL_SIZE,
							MAX(strs[2][5], strs[2][6]) / (long)FL_SIZE,
							&MD_ACCESS(D, strs[0], pos, (float*)args[0]),
							&MD_ACCESS(D, strs[1], pos, (float*)args[1]),
							&MD_ACCESS(D, strs[2], pos, (float*)args[2]));

		} while (md_next(D, dims[0], ~((MD_BIT(7) - 1) | MD_BIT(D - 1)), pos));

		return;
	}
#endif


	long slc_dims[D];
	md_select_dims(D, ~(MD_BIT(dim1) | MD_BIT(dim2)), slc_dims, dims[0]);

	long pos[D];
	md_set_dims(D, pos, 0);

	do {
		long offset = labs(upper_triag_idx(pos[dim1], pos[dim2])) * MAX(strs[2][dim1], strs[2][dim2]) / (long)FL_SIZE;

		md_fmac2(D, slc_dims, strs[0], &MD_ACCESS(D, strs[0], pos, (float*)args[0]), strs[1], &MD_ACCESS(D, strs[1], pos, (float*)args[1]), strs[2], (float*)args[2] + offset);

	} while (md_next(D, dims[0], MD_BIT(dim1) | MD_BIT(dim2), pos));
}


void md_tenmul_upper_triag2(int dim1, int dim2, int N, const long dims[N], const long ostrs[N], float* dst, const long istrs[N], const float* src, const long mdims[N], const long mstrs[N], const float* mat)
{
	md_clear2(N, dims, ostrs, dst, FL_SIZE);

	vptr_md_fmac_upper_triag2(dim1, dim2, 3, N, (const long*[3]) { dims, dims, mdims }, (const long*[3]) { ostrs, istrs, mstrs }, (void*[3]) { dst, (void*) src, (void*)mat });
}

void md_tenmul_upper_triag(int dim1, int dim2, int N, const long odims[N], float* dst, const long idims[N], const float* src, const long mdims[N], const float* mat)
{
	long msize = MAX(MAX(MAX(odims[dim1], odims[dim2]), idims[dim1]), idims[dim2]);

	assert(1 == odims[dim1] || msize == odims[dim1]);
	assert(1 == odims[dim2] || msize == odims[dim2]);
	assert(1 == idims[dim1] || msize == idims[dim1]);
	assert(1 == idims[dim2] || msize == idims[dim2]);
	assert(mdims[dim1] * mdims[dim2] == msize * (msize + 1) / 2);
	assert(1 == mdims[dim1] || 1 == mdims[dim2]);

	long max_dims[N];
	md_select_dims(N, ~(MD_BIT(dim1) | MD_BIT(dim2)), max_dims, mdims);

	assert(md_check_compat(N, ~0UL, odims, idims));
	assert(md_check_compat(N, ~0UL, max_dims, odims));
	assert(md_check_compat(N, ~0UL, max_dims, idims));

	md_max_dims(N, ~0UL, max_dims, max_dims, odims);
	md_max_dims(N, ~0UL, max_dims, max_dims, idims);

	long ostrs[N];
	md_calc_strides(N, ostrs, odims, FL_SIZE);

	long istrs[N];
	md_calc_strides(N, istrs, idims, FL_SIZE);

	long mstrs[N];
	md_calc_strides(N, mstrs, mdims, FL_SIZE);

	md_tenmul_upper_triag2(dim1, dim2, N, max_dims, ostrs, dst, istrs, src, mdims, mstrs, mat);
}




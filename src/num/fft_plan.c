#include <assert.h>
#include <complex.h>
#include <stdbool.h>

#ifndef NO_FFTW
#include <fftw3.h>
#else
#include <stdio.h>
typedef void *fftwf_plan;
#endif

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"


#include "misc/tree.h"
#include "misc/misc.h"
#include "misc/debug.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#include "fft-cuda.h"
#endif

#include "fft_plan.h"
#undef fft_plan_s



static tree_t fft_cache = NULL;

struct fft_plan_s {

	operator_data_t super;

	fftwf_plan fftw;

	int D;
	unsigned long flags;
	bool backwards;
	bool inplace;
	bool measure;
	const long* dims;
	const long* istrs;
	const long* ostrs;

#ifdef  USE_CUDA
	struct fft_cuda_plan_s* cuplan;
#endif
};

static DEF_TYPEID(fft_plan_s);

#define CMP(a, b) if (a != b) return (a > b) - (a < b);

static int fft_plan_cmp(const void* ptr1, const void* ptr2)
{
	const struct fft_plan_s* p1 = ptr1;
	const struct fft_plan_s* p2 = ptr2;

	CMP(p1->D, p2->D);
	CMP(p1->flags, p2->flags);
	CMP(p1->backwards, p2->backwards);
	CMP(p1->inplace, p2->inplace);
	CMP(p1->measure, p2->measure);

	for (int i = 0; i < p1->D; i++) {

		CMP(p1->dims[i], p2->dims[i]);
		CMP(p1->ostrs[i], p2->ostrs[i]);
		CMP(p1->istrs[i], p2->istrs[i]);
	}

	return 0;
}


static int fft_op_cmp(const void* _a, const void* _b)
{
	const struct operator_s* a = _a;
	const struct operator_s* b = _b;

	return fft_plan_cmp(CAST_DOWN(fft_plan_s, operator_get_data(a)), CAST_DOWN(fft_plan_s, operator_get_data(b)));
}

static int fft_op_plan_cmp(const void* _a, const void* b)
{
	const struct operator_s* a = _a;

	return fft_plan_cmp(CAST_DOWN(fft_plan_s, operator_get_data(a)), b);
}

void fft_cache_free(void)
{
	if (NULL == fft_cache)
		return;

	while (0 < tree_count(fft_cache))
		operator_free(tree_get_min(fft_cache, true));

	tree_free(fft_cache);
	fft_cache = NULL;
}

static struct operator_s* search(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], bool backwards, bool inplace, bool measure)
{
	if (NULL == fft_cache)
		return NULL;

	struct fft_plan_s ref = {

		.D = D,
		.flags = flags,
		.backwards = backwards,
		.inplace = inplace,
		.measure = measure,
		.dims = dimensions,
		.istrs = istrides,
		.ostrs = ostrides,
	};

	struct operator_s* op = NULL;

	op = tree_find(fft_cache, &ref, fft_op_plan_cmp, false);

	return op;
}


bool use_fftw_wisdom = false;

static char* fftw_wisdom_name(int N, bool backwards, unsigned long flags, const long dims[N])
{
	if (!use_fftw_wisdom)
		return NULL;

	const char* tbpath = getenv("BART_TOOLBOX_PATH");

	if (NULL == tbpath) {

		debug_printf(DP_WARN, "FFTW wisdom only works with BART_TOOLBOX_PATH set!\n");
		return NULL;
	}

	// Space for path and null terminator.
	int space = snprintf(NULL, 0, "%s/save/fftw/N_%d_BACKWARD_%d_FLAGS_%lu_DIMS", tbpath, N, backwards, flags);

	// Space for dimensions.
	for (int idx = 0; idx < N; idx ++)
		space += snprintf(NULL, 0, "_%lu", dims[idx]);

	// Space for extension.
	space += snprintf(NULL, 0, ".fftw");
	// Space for null terminator.
	space += 1;

	int len = space;
	char* loc = calloc((size_t)space, sizeof(char));

	if (NULL == loc)
		error("memory out\n");

	int ret = snprintf(loc, (size_t)len, "%s/save/fftw/N_%d_BACKWARD_%d_FLAGS_%lu_DIMS", tbpath, N, backwards, flags);

	assert(ret < len);
	len -= ret;

	for (int idx = 0; idx < N; idx++) {

		char tmp[64];
		ret = sprintf(tmp, "_%lu", dims[idx]);
		assert(ret < 64);
		len -= ret;
		strcat(loc, tmp);
	}

	strcat(loc, ".fftw");
	len -= 5;
	assert(1 == len);
	assert('\0' == loc[space - 1]);

	return loc;
}


#ifndef NO_FFTW
static fftwf_plan fft_fftwf_plan(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], complex float* dst, const long istrides[D], const complex float* src, bool backwards, bool measure)
{
	fftwf_plan fftwf;

	int N = D;
	fftwf_iodim64 dims[N];
	fftwf_iodim64 hmdims[N];
	int k = 0;
	int l = 0;

	char* wisdom = fftw_wisdom_name(D, backwards, flags, dimensions);

#pragma omp critical (bart_fftwf_plan)
	{
		if (NULL != wisdom)
			fftwf_import_wisdom_from_filename(wisdom);


		//FFTW seems to be fine with this
		//assert(0 != flags);

		for (int i = 0; i < N; i++) {

			if (MD_IS_SET(flags, i)) {

				dims[k].n = dimensions[i];
				dims[k].is = istrides[i] / (long)CFL_SIZE;
				dims[k].os = ostrides[i] / (long)CFL_SIZE;
				k++;

			} else  {

				hmdims[l].n = dimensions[i];
				hmdims[l].is = istrides[i] / (long)CFL_SIZE;
				hmdims[l].os = ostrides[i] / (long)CFL_SIZE;
				l++;
			}
		}
#ifndef NO_FFTW
		fftwf = fftwf_plan_guru64_dft(k, dims, l, hmdims, (complex float*)src, dst,
					backwards ? 1 : (-1), measure ? FFTW_MEASURE : FFTW_ESTIMATE);


		if (NULL != wisdom) {

			fftwf_export_wisdom_to_filename(wisdom);
			xfree(wisdom);
		}
#else
		assert(0);
#endif
	}

	return fftwf;
}
#endif


static void fft_apply(const operator_data_t* _plan, int N, void* args[N])
{
	complex float* dst = args[0];
	const complex float* src = args[1];
	const auto plan = CAST_DOWN(fft_plan_s, _plan);

	assert(2 == N);

	if (0u == plan->flags) {

		md_copy2(plan->D, plan->dims, plan->ostrs, dst, plan->istrs, src, CFL_SIZE);
		return;
	}

#ifdef  USE_CUDA
	if (cuda_ondevice(src)) {

#pragma 	omp critical(cufft_create_plan_in_threads)
		if (NULL == plan->cuplan)
			plan->cuplan = fft_cuda_plan(plan->D, plan->dims, plan->flags, plan->ostrs, plan->istrs, plan->backwards);

		if (NULL == plan->cuplan)
			error("Failed to plan a GPU FFT (too large?)\n");

		fft_cuda_exec(plan->cuplan, dst, src);

	} else
#endif
	{
		assert(NULL != plan->fftw);
#ifndef NO_FFTW
		fftwf_execute_dft(plan->fftw, (complex float*)src, dst);
#endif
	}
}


static void fft_free_plan(const operator_data_t* _data)
{
	const auto plan = CAST_DOWN(fft_plan_s, _data);
#ifndef NO_FFTW
	if (NULL != plan->fftw)
		fftwf_destroy_plan(plan->fftw);
#endif

#ifdef	USE_CUDA
	if (NULL != plan->cuplan)
		fft_cuda_free_plan(plan->cuplan);
#endif
	xfree(plan->dims);
	xfree(plan->istrs);
	xfree(plan->ostrs);

	xfree(plan);
}


const struct operator_s* fft_create2(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const complex float* dst, const long istrides[D], const complex float* src, bool backwards)
{
	flags &= md_nontriv_dims(D, dimensions);

	long ooffset = 0;
	long ioffset = 0;

	long osize = CFL_SIZE;
	long isize = CFL_SIZE;

	for (int i = 0; i < D; i++) {

		osize += (dimensions[i] - 1) * labs(ostrides[i]);
		isize += (dimensions[i] - 1) * labs(istrides[i]);

		ooffset += (dimensions[i] - 1) * MAX(-ostrides[i], 0);
		ioffset += (dimensions[i] - 1) * MAX(-istrides[i], 0);
	}

	const complex float* srcs = src - ioffset / (long)CFL_SIZE;
	const complex float* srce = srcs + isize / (long)CFL_SIZE;

	const complex float* dsts = dst - ioffset / (long)CFL_SIZE;
	const complex float* dste = dsts + isize / (long)CFL_SIZE;

	bool inplace;

	if ((dsts > srce) || (srcs > dste))
		inplace = false;
	else
		inplace = true;

	bool trivial =    (D == md_calc_blockdim(D, dimensions, ostrides, CFL_SIZE))
		       && (D == md_calc_blockdim(D, dimensions, istrides, CFL_SIZE));

	bool measure = trivial && use_fftw_wisdom;

	const struct operator_s* op;

#pragma omp critical(bart_fftcache)
	{
		op = search(D, dimensions, flags, ostrides, istrides, backwards, inplace, measure);

		if (NULL != op) {

			op = operator_ref(op);
		} else {

			long size = MAX(isize, osize);

			complex float* tsrc = md_alloc(1, MD_DIMS(size), 1);
			complex float* tdst = inplace ? tsrc : md_alloc(1, MD_DIMS(size), 1);

			PTR_ALLOC(struct fft_plan_s, plan);
			SET_TYPEID(fft_plan_s, plan);

			plan->fftw = NULL;

#ifndef NO_FFTW
			if (0u != flags)
				plan->fftw = fft_fftwf_plan(D, dimensions, flags, ostrides, tdst + ooffset, istrides, tsrc + ioffset, backwards, use_fftw_wisdom && trivial);
#endif

			md_free(tsrc);
			if (!inplace)
				md_free(tdst);


#ifdef  USE_CUDA
			plan->cuplan = NULL;
#endif
			plan->D = D;
			plan->flags = flags;
			plan->backwards = backwards;
			plan->inplace = inplace;
			plan->measure = measure;

			PTR_ALLOC(long[D], dims);
			md_copy_dims(D, *dims, dimensions);
			plan->dims = *PTR_PASS(dims);

			PTR_ALLOC(long[D], istrs);
			md_copy_strides(D, *istrs, istrides);
			plan->istrs = *PTR_PASS(istrs);

			PTR_ALLOC(long[D], ostrs);
			md_copy_strides(D, *ostrs, ostrides);
			plan->ostrs = *PTR_PASS(ostrs);

			op = operator_create2(D, dimensions, ostrides, D, dimensions, istrides, CAST_UP(PTR_PASS(plan)), fft_apply, fft_free_plan);

			if (NULL == fft_cache)
				fft_cache = tree_create(fft_op_cmp);

			tree_insert(fft_cache, (void*)operator_ref(op));
		}
	}

	return op;
}


void fft_set_num_threads(int n)
{
	static bool fft_threads_init = false;

#ifdef FFTWTHREADS
#pragma omp critical (bart_fftwf_plan)
	if (!fft_threads_init) {

		fft_threads_init = true;
		fftwf_init_threads();
	}

#pragma omp critical (bart_fftwf_plan)
        fftwf_plan_with_nthreads(n);
#endif
}



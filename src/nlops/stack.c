#include <complex.h>

#include "misc/types.h"
#include "misc/misc.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/cast.h"

#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "stack.h"


struct stack_s {

	INTERFACE(nlop_data_t);

	int N;
	const long* idims1;
	const long* idims2;
	const long* odims;

	const long* istrs1;
	const long* istrs2;
	const long* ostrs;

	long offset;
};

DEF_TYPEID(stack_s);


static void stack_fun(const nlop_data_t* _data, int N, complex float* args[N])
{
	const auto data = CAST_DOWN(stack_s, _data);
	assert(3 == N);

	complex float* dst = args[0];
	const complex float* src1 = args[1];
	const complex float* src2 = args[2];

#ifdef USE_CUDA
	assert((cuda_ondevice(dst) == cuda_ondevice(src1)) && (cuda_ondevice(src1) == cuda_ondevice(src2)));
#endif

	md_copy2(data->N, data->idims1, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst, MD_STRIDES(data->N, data->idims1, CFL_SIZE), src1, CFL_SIZE);
	md_copy2(data->N, data->idims2, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst + data->offset, MD_STRIDES(data->N, data->idims2, CFL_SIZE), src2, CFL_SIZE);
}

static void stack_der2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(stack_s, _data);
	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_copy2(data->N, data->idims2, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst + data->offset, MD_STRIDES(data->N, data->idims2, CFL_SIZE), src, CFL_SIZE);
}

static void stack_adj2(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(stack_s, _data);
	md_copy2(data->N, data->idims2, MD_STRIDES(data->N, data->idims2, CFL_SIZE) , dst, MD_STRIDES(data->N, data->odims, CFL_SIZE), src + data->offset, CFL_SIZE);
}

static void stack_der1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);


	const auto data = CAST_DOWN(stack_s, _data);
	md_clear(data->N, data->odims, dst, CFL_SIZE);
	md_copy2(data->N, data->idims1, MD_STRIDES(data->N, data->odims, CFL_SIZE), dst, MD_STRIDES(data->N, data->idims1, CFL_SIZE), src, CFL_SIZE);
}

static void stack_adj1(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto data = CAST_DOWN(stack_s, _data);
	md_copy2(data->N, data->idims1, MD_STRIDES(data->N, data->idims1, CFL_SIZE), dst, MD_STRIDES(data->N, data->odims, CFL_SIZE), src, CFL_SIZE);
}


static void stack_del(const nlop_data_t* _data)
{
	const auto data = CAST_DOWN(stack_s, _data);

	xfree(data->idims1);
	xfree(data->idims2);
	xfree(data->odims);

	xfree(data->istrs1);
	xfree(data->istrs2);
	xfree(data->ostrs);

	xfree(data);
}


struct nlop_s* nlop_stack_create(int N, const long odims[N], const long idims1[N], const long idims2[N], unsigned long stack_dim)
{
	assert((int)stack_dim < N);

	PTR_ALLOC(struct stack_s, data);
	SET_TYPEID(stack_s, data);

	data->offset = 1;

	for (unsigned int i = 0; i < (unsigned)N; i++)
	{
		if (i == stack_dim)
			assert(odims[i] == (idims1[i] + idims2[i]));
		else
			assert((odims[i] == idims1[i]) && (odims[i] == idims2[i]));

		if (i <= stack_dim)
			data->offset *= idims1[i];

	}

	PTR_ALLOC(long[N], nodims);
	PTR_ALLOC(long[N], nidims1);
	PTR_ALLOC(long[N], nidims2);
	md_copy_dims(N, *nodims, odims);
	md_copy_dims(N, *nidims1, idims1);
	md_copy_dims(N, *nidims2, idims2);

	PTR_ALLOC(long[N], nistr1);
	md_calc_strides(N, *nistr1, idims1, CFL_SIZE);
	data->istrs1 = *PTR_PASS(nistr1);

	PTR_ALLOC(long[N], nistr2);
	md_calc_strides(N, *nistr2, idims1, CFL_SIZE);
	data->istrs2 = *PTR_PASS(nistr2);

	PTR_ALLOC(long[N], nostr);
	md_calc_strides(N, *nostr, odims, CFL_SIZE);
	data->ostrs = *PTR_PASS(nostr);

	data->N = N;
	data->odims = *PTR_PASS(nodims);
	data->idims1 = *PTR_PASS(nidims1);
	data->idims2 = *PTR_PASS(nidims2);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->odims);


	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], data->idims1);
	md_copy_dims(N, nl_idims[1], data->idims2);


	return nlop_generic_create(1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
		stack_fun, (nlop_der_fun_t[2][1]){ { stack_der1 }, { stack_der2 } }, (nlop_der_fun_t[2][1]){ { stack_adj1 }, { stack_adj2 } }, NULL, NULL, stack_del);
}


struct nlop_s* nlop_destack_create(int N, const long odims1[N], const long odims2[N], const long idims[N], unsigned long stack_dim)
{
	assert((int)stack_dim < N);

	for (unsigned int i = 0; i < (unsigned)N; i++)
	{
		if (i == stack_dim)
			assert(odims1[i] + odims2[i] == idims[i]);
		else
			assert((odims1[i] == idims[i]) && (odims1[i] == odims2[i]));
	}

	long pos1[N];
	long pos2[N];
	for(int i = 0; i < N; i++) {

		pos1[i] = 0;
		pos2[i] = ((unsigned long)i == stack_dim) ? odims1[i] : 0;
	}

	auto extract1 = nlop_from_linop_F(linop_extract_create(N, pos1, odims1, idims));
	auto extract2 = nlop_from_linop_F(linop_extract_create(N, pos2, odims2, idims));

	auto combined = nlop_combine_FF(extract1, extract2);
	return nlop_dup_F(combined, 0, 1);
}
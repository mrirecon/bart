
#include <stdlib.h>
#include <assert.h>
#include <complex.h>

#include "num/ops.h"
#include "num/iovec.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/shrdptr.h"
#include "misc/misc.h"
#include "misc/debug.h"

#include "ops_p.h"


// copy from num/ops.c
struct operator_s {

	unsigned int N;
	unsigned int io_flags;
	const struct iovec_s** domain;

	operator_data_t* data;
	void (*apply)(const operator_data_t* data, unsigned int N, void* args[N]);
	void (*del)(const operator_data_t* data);

	struct shared_obj_s sptr;
};



struct operator_p_s {

//	struct operator_s op;
};

const struct operator_p_s* operator_p_ref(const struct operator_p_s* x)
{
	if (NULL != x)
		operator_ref((const struct operator_s*)x);

	return x;
}

/**
 * Return the dimensions and strides of the domain of an operator_p
 *
 * @param op operator_p
 */
const struct iovec_s* operator_p_domain(const struct operator_p_s* _op)
{
	auto op = (const struct operator_s*)_op;
	assert(3 == op->N);
	assert(2u == op->io_flags);
	return op->domain[2];
}


/**
 * Return the dimensions and strides of the codomain of an operator_p
 *
 * @param op operator_p
 */
const struct iovec_s* operator_p_codomain(const struct operator_p_s* _op)
{
	auto op = (const struct operator_s*)_op;
	assert(3 == op->N);
	assert(2u == op->io_flags);
	return op->domain[1];
}



void operator_p_free(const struct operator_p_s* x)
{
	if (NULL != x)
		operator_free((const struct operator_s*)x);
}


struct op_p_data_s {

	INTERFACE(operator_data_t);

	operator_data_t* data;
	operator_p_fun_t apply;
	operator_del_t del;
};

static DEF_TYPEID(op_p_data_s);

static void op_p_apply(const operator_data_t* _data, unsigned int N, void* args[N])
{
	const auto data = CAST_DOWN(op_p_data_s, _data);
	assert(3 == N);	// FIXME: gpu
	data->apply(data->data, *((float*)args[0]), args[1], args[2]);
}

static void op_p_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(op_p_data_s, _data);
	data->del(data->data);
	xfree(data);
}

operator_data_t* operator_p_get_data(const struct operator_p_s* _data)
{
	const auto data = CAST_DOWN(op_p_data_s, operator_get_data(operator_p_upcast(_data)));
	return data->data;
}




static void operator_del(const struct shared_obj_s* sptr)
{
	const struct operator_s* x = CONTAINER_OF(sptr, const struct operator_s, sptr);

	if (NULL != x->del)
		x->del(x->data);

	for (unsigned int i = 0; i < x->N; i++)
		iovec_free(x->domain[i]);

	xfree(x->domain);
	xfree(x);
}



/**
 * Create an operator with one parameter (without strides)
 */
const struct operator_p_s* operator_p_create2(unsigned int ON, const long out_dims[ON], const long out_strs[ON], 
		unsigned int IN, const long in_dims[IN], const long in_strs[IN],
		operator_data_t* data, operator_p_fun_t apply, operator_del_t del)
{
	PTR_ALLOC(struct operator_s, o);
	PTR_ALLOC(struct op_p_data_s, op);
	SET_TYPEID(op_p_data_s, op);

	op->data = data;
	op->apply = apply;
	op->del = del;

	PTR_ALLOC(const struct iovec_s*[3], dom);

	(*dom)[0] = iovec_create2(1, MD_DIMS(1), MD_DIMS(0), FL_SIZE);
	(*dom)[1] = iovec_create2(ON, out_dims, out_strs, CFL_SIZE);
	(*dom)[2] = iovec_create2(IN, in_dims, in_strs, CFL_SIZE);

	o->N = 3;
	o->io_flags = MD_BIT(1);
	o->domain = *PTR_PASS(dom);
	o->data = CAST_UP(PTR_PASS(op));
	o->apply = op_p_apply;
	o->del = op_p_del;

	shared_obj_init(&o->sptr, operator_del);

	if (NULL == del)
		debug_printf(DP_WARN, "Warning: no delete function specified for operator_p_create! Possible memory leak.\n");

	return operator_p_downcast(PTR_PASS(o));
}


/**
 * Create an operator with one parameter (without strides)
 *
 * @param ON number of output dimensions
 * @param out_dims dimensions of output
 * @param IN number of input dimensions
 * @param in_dims dimensions of input
 * @param data data for applying the operation
 * @param apply function that applies the operation
 * @param del function that frees the data
 */
const struct operator_p_s* operator_p_create(unsigned int ON, const long out_dims[ON], 
		unsigned int IN, const long in_dims[IN], 
		operator_data_t* data, operator_p_fun_t apply, operator_del_t del)
{
	return operator_p_create2(ON, out_dims, MD_STRIDES(ON, out_dims, CFL_SIZE),
				IN, in_dims, MD_STRIDES(IN, in_dims, CFL_SIZE),
				data, apply, del);
}



const struct operator_s* operator_p_upcast(const struct operator_p_s* op)
{
	return (const struct operator_s*)op;
}

const struct operator_p_s* operator_p_downcast(const struct operator_s* op)
{
	assert(3 == op->N);
	assert(2u == op->io_flags);

	return (const struct operator_p_s*)op;
}



const struct operator_p_s* operator_p_pre_chain(const struct operator_s* a, const struct operator_p_s* _b)
{
	auto b = operator_p_upcast(_b);

	assert((2 == a->N) && (3 == b->N));

	const struct operator_s* x = operator_combi_create(2, MAKE_ARRAY(b, a));
	const struct operator_s* y = operator_link_create(x, 3, 2);	// mu bo bi a0 ai

	operator_free(x);

	return operator_p_downcast(y);
}

const struct operator_p_s* operator_p_pst_chain(const struct operator_p_s* _a, const struct operator_s* b)
{
	const struct operator_s* a = (const struct operator_s*)_a;

	assert((3 == a->N) && (2 == b->N));

	const struct operator_s* x = operator_combi_create(2, MAKE_ARRAY(b, a));
	const struct operator_s* y = operator_link_create(x, 3, 1);	// bo bi, mu a0 ai
	const struct operator_s* z = operator_permute(y, 3, (int[]){ 1, 0, 2 });

	operator_free(x);
	operator_free(y);

	return operator_p_downcast(z);
}

void operator_p_apply2(const struct operator_p_s* _op, float mu, unsigned int ON, const long odims[ON], const long ostrs[ON], complex float* dst, const long IN, const long idims[IN], const long istrs[IN], const complex float* src)
{
	auto op = operator_p_upcast(_op);

	assert(3 == op->N);
	assert(iovec_check(op->domain[2], IN, idims, istrs));
	assert(iovec_check(op->domain[1], ON, odims, ostrs));

	operator_p_apply_unchecked(_op, mu, dst, src);
}


void operator_p_apply(const struct operator_p_s* op, float mu, unsigned int ON, const long odims[ON], complex float* dst, const long IN, const long idims[IN], const complex float* src)
{
	operator_p_apply2(op, mu,
			ON, odims, MD_STRIDES(ON, odims, CFL_SIZE), dst,
			IN, idims, MD_STRIDES(IN, idims, CFL_SIZE), src);
}


void operator_p_apply_unchecked(const struct operator_p_s* _op, float mu, complex float* dst, const complex float* src)
{
	auto op = operator_p_upcast(_op);

	assert(3 == op->N);
	op->apply(op->data, 3, (void*[3]){ &mu, (void*)dst, (void*)src });
}


const struct operator_s* operator_p_bind(const struct operator_p_s* op, float alpha)
{
	float* nalpha = xmalloc(sizeof(float));
	*nalpha = alpha;

	return operator_attach(operator_bind2(operator_p_upcast(op), 0, 1, (long[]){ 1 }, (long[]){ 0 }, nalpha), nalpha, xfree);
}



const struct operator_p_s* operator_p_gpu_wrapper(const struct operator_p_s* op)
{
	return operator_p_downcast(operator_gpu_wrapper2(operator_p_upcast(op), MD_BIT(1) | MD_BIT(2)));
}


const struct operator_p_s* operator_p_stack(int A, int B, const struct operator_p_s* _a, const struct operator_p_s* _b)
{
	auto a = operator_p_upcast(_a);
	auto b = operator_p_upcast(_b);

	auto c = operator_stack2(2, (int[]){ 1, 2 }, (int[]){ A, B }, a, b);

	return operator_p_downcast(c);
}


struct scale_s {

	INTERFACE(operator_data_t);

	long size;
};

DEF_TYPEID(scale_s);

static void op_p_scale_apply(const operator_data_t* _data, float mu, complex float* dst, const complex float* src)
{
	auto data = CAST_DOWN(scale_s, _data);

	md_zsmul(1, MD_DIMS(data->size), dst, src, mu);
}

static void op_p_scale_del(const operator_data_t* _data)
{
	xfree(CAST_DOWN(scale_s, _data));
}

const struct operator_p_s* operator_p_scale(int N, const long dims[N])
{
	PTR_ALLOC(struct scale_s, data);
	SET_TYPEID(scale_s, data);

	data->size = md_calc_size(N, dims);

	return operator_p_create(N, dims, N, dims, CAST_UP(PTR_PASS(data)), op_p_scale_apply, op_p_scale_del);
}



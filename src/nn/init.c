/* Copyright 2020-2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: Moritz Blumenthal
 */

#include <stdbool.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/shrdptr.h"
#include "misc/types.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/rand.h"
#include "init.h"

typedef void (*initializer_del)(const struct initializer_s* conf);

typedef struct initializer_s {

	TYPEID* TYPEID;

	initializer_f fun;
	initializer_del del;
	struct shared_obj_s sptr;

} init_t;

static void init_del(const struct shared_obj_s* sptr)
{
	const struct initializer_s* x = CONTAINER_OF(sptr, const struct initializer_s, sptr);

	if (NULL != x->del)
		x->del(x);

	xfree(x);
}

void initializer_free(const struct initializer_s* x)
{
	if (NULL == x)
		return;

	shared_obj_destroy(&x->sptr);
}

const struct initializer_s* initializer_clone(const struct initializer_s* x)
{
	if (NULL != x)
		shared_obj_ref(&x->sptr);

	return x;
}

void initializer_apply(const struct initializer_s* x, long N, const long dims[N], complex float* weights)
{
	x->fun(x, N, dims, weights);
}

unsigned long in_flag_conv_generic(int N, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag)
{
	unsigned long in_flag = 0;
	for (int i = N - 1; i >= 0; i--){

		if (MD_IS_SET(conv_flag, i)) {

			in_flag = MD_SET(in_flag, i);
			continue;
		}

		if (MD_IS_SET(channel_flag, i)){

			in_flag = MD_SET(in_flag, i);
			in_flag *= 2;
			continue;
		}

		if (MD_IS_SET(group_flag, i))
			continue;

		in_flag /= 2;
	}

	return in_flag;
}

unsigned long out_flag_conv_generic(int N, unsigned long conv_flag, unsigned long channel_flag, unsigned long group_flag)
{
	unsigned long out_flag = 0;

	for (int i = N - 1; i >= 0; i--) {

		if (MD_IS_SET(conv_flag, i)) {

			out_flag = MD_SET(out_flag, i);

			continue;
		}

		if (MD_IS_SET(channel_flag, i)) {

			out_flag *= 2;
			out_flag = MD_SET(out_flag, i);

			continue;
		}

		if (MD_IS_SET(group_flag, i))
			continue;

		out_flag /= 2;
	}

	return out_flag;
}

unsigned long in_flag_conv(bool c1)
{
	unsigned long in_flags = c1 ? MD_BIT(1) : MD_BIT(3);

	//filters, channel, kx, ky, kz    or x, y, z channel, filters
	for (int i = 0; i < 3; i++)
		in_flags |= MD_BIT(i + (c1 ? 0 : 2));

	return in_flags;
}

unsigned long out_flag_conv(bool c1)
{
	unsigned long out_flags = c1 ? MD_BIT(0) : MD_BIT(4);

	//filters, channel, kx, ky, kz    or x, y, z channel, filters
	for (int i = 0; i < 3; i++)
		out_flags |= MD_BIT(i + (c1 ? 0 : 2));

	return out_flags;
}

struct initializer_const_s {

	INTERFACE(init_t);
	complex float val;
};

static DEF_TYPEID(initializer_const_s);

static void init_const_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_const_s, conf_);

	md_zfill(N, dims, weights, conf->val);
}

/**
 * Create a constant initializer
 *
 * @param val value to initialize weights with
 *
 * @returns Constant initializer
 */
const struct initializer_s* init_const_create(_Complex float val)
{
	PTR_ALLOC(struct initializer_const_s, data);
	SET_TYPEID(initializer_const_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_const_fun;

	data->val = val;

	return CAST_UP(PTR_PASS(data));
}

struct initializer_fixed_s {

	INTERFACE(init_t);

	int N;
	const long* dims;

	complex float* data;
};

static DEF_TYPEID(initializer_fixed_s);

static void init_fixed_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_fixed_s, conf_);

	assert(N == conf->N);
	assert(md_check_equal_dims(N, dims, conf->dims, ~0));

	md_copy(N, dims, weights, conf->data, CFL_SIZE);
}

static void init_fixed_del(const init_t* conf_)
{
	auto d = CAST_DOWN(initializer_fixed_s, conf_);

	md_free(d->data);
	xfree(d->dims);
}

/**
 * Create a constant initializer from array
 *
 * @returns Constant initializer
 */
const struct initializer_s* init_array_create(int N, const long dims[N], const complex float* dat)
{
	PTR_ALLOC(struct initializer_fixed_s, data);
	SET_TYPEID(initializer_fixed_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);

	data->INTERFACE.del = init_fixed_del;
	data->INTERFACE.fun = init_fixed_fun;

	data->N = N;
	data->dims = ARR_CLONE(long[N], dims);

	complex float* tmp = md_alloc(N, dims, CFL_SIZE);
	md_copy(N, dims, tmp, dat, CFL_SIZE);

	data->data = tmp;

	return CAST_UP(PTR_PASS(data));
}

// Returns real/complex uniform/normal distribution with mean 0 and variance 1
static void get_base_dist(unsigned int N, const long dims[N], complex float* dst, bool uniform, bool real)
{
	if (uniform) {

		md_uniform_rand(N, dims, dst);
		md_zsadd(N, dims, dst, dst, (complex float)(-0.5));

		if (!real) {

			complex float* tmp = md_alloc_sameplace(N, dims, CFL_SIZE, dst);

			md_uniform_rand(N, dims, tmp);
			md_zsadd(N, dims, tmp, tmp, (complex float)(-0.5));
			md_zaxpy(N, dims, dst, 1.I, tmp);

			md_free(tmp);
		}

		md_zsmul(N, dims, dst, dst, real ? sqrt(12) : sqrt(6));

	} else {

		md_gaussian_rand(N, dims, dst);

		if (real)
			md_zreal(N, dims, dst, dst);
		else
			md_zsmul(N, dims, dst, dst, 1. / sqrt(2.));
	}
}

/*
Xavier Glorot, Yoshua Bengio ; Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, JMLR Workshop and Conference Proceedings 9:249-256, 2010.
Glorot, X. & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, in PMLR 9:249-256
*/
static float get_scaling_xavier(unsigned int N, const long dims[N], unsigned long in_flags, unsigned long out_flags)
{
	long tdims[N];
	md_select_dims(N, in_flags, tdims, dims);

	long inputs = md_calc_size(N, tdims);

	md_select_dims(N, out_flags, tdims, dims);
	long outputs = md_calc_size(N, tdims);

	return (float)sqrt(2. / (double)(inputs + outputs));
}

/*
He, K.; Zhang, X.; Ren, S. & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
*/
static float get_scaling_kaiming(unsigned int N, const long dims[N], unsigned long in_flags, float leaky_val)
{
	long tdims[N];
	md_select_dims(N, in_flags, tdims, dims);
	long inputs = md_calc_size(N, tdims);

	return (float)sqrt(2. / (double)(inputs) / (1. + leaky_val * leaky_val));
}

struct initializer_xavier_kaiming_s {

	INTERFACE(init_t);

	bool uniform;
	bool real;

	unsigned long in_flags;
	unsigned long out_flags;

	float leaky_val;
};

static DEF_TYPEID(initializer_xavier_kaiming_s);

static void init_xavier_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_xavier_kaiming_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, get_scaling_xavier(N, dims, conf->in_flags, conf->out_flags));
}

static void init_kaiming_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_xavier_kaiming_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, get_scaling_kaiming(N, dims, conf->in_flags, conf->leaky_val));
}


/**
 * Create a Xavier (Glorot) initializer
 *
 * Xavier Glorot, Yoshua Bengio ; Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, JMLR Workshop and Conference Proceedings 9:249-256, 2010.
 * Glorot, X. & Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, in PMLR 9:249-256
 *
 * Initializes weights with random numbers drawn from a uniform/normal distribution with standard deviation sqrt(2 / (inputs + outputs))
 *
 * @param in_flags bitmask selecting the dimensions corresponding to inputs
 * @param out_flags bitmask selecting the dimensions corresponding to outputs
 * @param real if true: the imaginary part is initialized with zeros
 * @param uniform if true: the distribution is a scaled uniform distribution; else: the distribution is a scaled normal distribution
 *
 * @returns Xavier initializer
 */
const struct initializer_s* init_xavier_create(unsigned long in_flags, unsigned long out_flags, bool real, bool uniform)
{
	PTR_ALLOC(struct initializer_xavier_kaiming_s, data);
	SET_TYPEID(initializer_xavier_kaiming_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_xavier_fun;

	data->in_flags = in_flags;
	data->out_flags = out_flags;
	data->uniform = uniform;
	data->real = real;
	data->leaky_val = 1;

	return CAST_UP(PTR_PASS(data));
}

/**
 * Create a Kaiming initializer
 *
 * He, K.; Zhang, X.; Ren, S. & Sun, J. (2015). Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
 *
 * Initializes weights with random numbers drawn from a uniform/normal distribution with standard deviation sqrt(2 / (inputs *(1 + leaky_val^2)))
 *
 * @param in_flags bitmask selecting the dimensions corresponding to inputs
 * @param real if true: the imaginary part is initialized with zeros
 * @param uniform if true: the distribution is a scaled uniform distribution; else: the distribution is a scaled normal distribution
 * @param leaky_val initialization Parametric ReLU / Leaky ReLU
 *
 * @returns Kaiming initializer
 */
const struct initializer_s* init_kaiming_create(unsigned long in_flags, bool real, bool uniform, float leaky_val)
{
	PTR_ALLOC(struct initializer_xavier_kaiming_s, data);
	SET_TYPEID(initializer_xavier_kaiming_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_kaiming_fun;

	data->in_flags = in_flags;
	data->out_flags = 0;
	data->uniform = uniform;
	data->real = real;
	data->leaky_val = leaky_val;

	return CAST_UP(PTR_PASS(data));
}

struct initializer_std_normal_s {

	INTERFACE(init_t);

	bool uniform;
	bool real;

	float scale;
	float mean;
};

static DEF_TYPEID(initializer_std_normal_s);

static void init_std_normal_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_std_normal_s, conf_);

	get_base_dist(N, dims, weights, conf->uniform, conf->real);
	md_zsmul(N, dims, weights, weights, conf->scale);
	md_zsadd(N, dims, weights, weights, conf->mean + (conf->real ? 0 : I * conf->mean));
}

const struct initializer_s* init_std_normal_create(bool real, float scale, float mean)
{
	PTR_ALLOC(struct initializer_std_normal_s, data);
	SET_TYPEID(initializer_std_normal_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_std_normal_fun;

	data->uniform = false;
	data->real = real;
	data->scale = scale;
	data->mean = mean;

	return CAST_UP(PTR_PASS(data));
}

const struct initializer_s* init_uniform_create(bool real, float scale, float mean)
{
	PTR_ALLOC(struct initializer_std_normal_s, data);
	SET_TYPEID(initializer_std_normal_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_std_normal_fun;

	data->uniform = true;
	data->real = real;
	data->scale = scale;
	data->mean = mean;

	return CAST_UP(PTR_PASS(data));
}

struct initializer_linspace_s {

	INTERFACE(init_t);

	unsigned int dim;

	complex float min_val;
	complex float max_val;

	bool max_inc;
};

static DEF_TYPEID(initializer_linspace_s);

static void init_linspace_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto conf = CAST_DOWN(initializer_linspace_s, conf_);

	assert(conf->dim < N);

	complex float vals[dims[conf->dim]];
	for (int i = 0; i < dims[conf->dim]; i++)
		vals[i] = conf->min_val + i *(conf->max_val - conf->min_val) / ((float)dims[conf->dim] - (conf->max_inc? 1. : 0));

	long vdims[N];
	md_select_dims(N, MD_BIT(conf->dim), vdims, dims);

	md_copy2(N, dims, MD_STRIDES(N, dims, CFL_SIZE), weights, MD_STRIDES(N, vdims, CFL_SIZE), vals, CFL_SIZE);
}

/**
 * Create a initializer to initialize with linear spacing along a selected dimension and constant along the other
 *
 * @param dim dimension along which the values are changing
 * @param min_val
 * @param max_val
 * @param max_inc should the maximal value be included
 *
 * @returns Linear spaced initializer
 */
const struct initializer_s* init_linspace_create(unsigned int dim, complex float min_val, complex float max_val, bool max_inc)
{
	PTR_ALLOC(struct initializer_linspace_s, data);
	SET_TYPEID(initializer_linspace_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = NULL;
	data->INTERFACE.fun = init_linspace_fun;

	data->dim = dim;
	data->max_val = max_val;
	data->max_inc = max_inc;
	data->min_val = min_val;

	return CAST_UP(PTR_PASS(data));
}

struct initializer_reshape_s {

	INTERFACE(init_t);

	unsigned int N;
	long* dims;

	const struct initializer_s* init;
};

static DEF_TYPEID(initializer_reshape_s);

static void init_reshape_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto d = CAST_DOWN(initializer_reshape_s, conf_);

	assert(md_calc_size(N, dims) == md_calc_size(d->N, d->dims));

	initializer_apply(d->init, d->N, d->dims, weights);
}

static void init_reshape_del(const init_t* conf_)
{
	auto d = CAST_DOWN(initializer_reshape_s, conf_);
	initializer_free(d->init);
	xfree(d->dims);
}

/**
 * Used internally to apply initializer with original dimensions if the input of a nn_t is reshaped
 */
const struct initializer_s* init_reshape_create(unsigned int N, const long dims[N], const struct initializer_s* init)
{
	if(NULL == init)
		return NULL;
	PTR_ALLOC(struct initializer_reshape_s, data);
	SET_TYPEID(initializer_reshape_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = init_reshape_del;
	data->INTERFACE.fun = init_reshape_fun;

	data->N = N;
	PTR_ALLOC(long[N], ndims);
	md_copy_dims(N, *ndims, dims);
	data->dims = *PTR_PASS(ndims);
	data->init = initializer_clone(init);

	return CAST_UP(PTR_PASS(data));
}

struct initializer_stack_s {

	INTERFACE(init_t);

	unsigned int N;
	long* dims;
	long* dimsa;
	long* dimsb;
	int stack_dim;

	const struct initializer_s* inita;
	const struct initializer_s* initb;
};

static DEF_TYPEID(initializer_stack_s);

static void init_stack_fun(const init_t* conf_, long N, const long dims[N], complex float* weights)
{
	auto d = CAST_DOWN(initializer_stack_s, conf_);

	assert(md_calc_size(N, dims) == md_calc_size(d->N, d->dimsa) + md_calc_size(d->N, d->dimsb));

	complex float* weightsa = md_alloc(d->N, d->dimsa, CFL_SIZE);
	complex float* weightsb = md_alloc(d->N, d->dimsb, CFL_SIZE);

	initializer_apply(d->inita, d->N, d->dimsa, weightsa);
	initializer_apply(d->initb, d->N, d->dimsb, weightsb);

	long pos[d->N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	md_copy_block(d->N, pos, d->dims, weights, d->dimsa, weightsa, CFL_SIZE);
	pos[d->stack_dim] = d->dimsa[d->stack_dim];
	md_copy_block(d->N, pos, d->dims, weights, d->dimsb, weightsb, CFL_SIZE);

	md_free(weightsa);
	md_free(weightsb);
}

static void init_stack_del(const init_t* conf_)
{
	auto d = CAST_DOWN(initializer_stack_s, conf_);
	initializer_free(d->inita);
	initializer_free(d->initb);
	xfree(d->dims);
	xfree(d->dimsa);
	xfree(d->dimsb);
}

/**
 * Used internally to apply initializers of original inputs if two inputs of a nn_t are stacked
 * If only one initializer is set, the other will fall back to a zero initializer
 */
const struct initializer_s* init_stack_create(unsigned int N, int stack_dim, const long dimsa[N], const struct initializer_s* inita, const long dimsb[N], const struct initializer_s* initb)
{
	if (NULL == inita && NULL == initb)
		return NULL;

	PTR_ALLOC(struct initializer_stack_s, data);
	SET_TYPEID(initializer_stack_s, data);

	shared_obj_init(&(data->INTERFACE.sptr), init_del);
	data->INTERFACE.del = init_stack_del;
	data->INTERFACE.fun = init_stack_fun;

	data->N = N;

	data->stack_dim = (0 > stack_dim) ? (int)N + stack_dim : stack_dim;
	assert((0 <= data->stack_dim) && (data->stack_dim < (int)N));

	data->inita = (NULL == inita) ? init_const_create(0) : initializer_clone(inita);
	data->initb = (NULL == initb) ? init_const_create(0) : initializer_clone(initb);

	PTR_ALLOC(long[N], ndimsa);
	md_copy_dims(N, *ndimsa, dimsa);
	data->dimsa = *PTR_PASS(ndimsa);

	PTR_ALLOC(long[N], ndimsb);
	md_copy_dims(N, *ndimsb, dimsb);
	data->dimsb = *PTR_PASS(ndimsb);

	PTR_ALLOC(long[N], dims);
	for (int i = 0; i < (int)N; i++) {

		if (i == data->stack_dim) {

			(*dims)[i] = dimsa[i] + dimsb[i];
		} else {
			assert(dimsa[i] == dimsb[i]);
			(*dims)[i] = dimsa[i];
		}
	}
	data->dims = *PTR_PASS(dims);

	return CAST_UP(PTR_PASS(data));
}

const struct initializer_s* init_dup_create(const struct initializer_s* inita, const struct initializer_s* initb)
{
	if (NULL == inita && NULL == initb)
		return NULL;

	if ((NULL == inita) && (NULL != initb))
		return initializer_clone(initb);

	if ((NULL == initb) && (NULL != inita))
		return initializer_clone(inita);

	if (inita->TYPEID != initb->TYPEID)
		error("Dup for arguments with different initializers, i.e. \"%s\" and \"%s\"!", inita->TYPEID->name, initb->TYPEID->name);
	return initializer_clone(inita);
}

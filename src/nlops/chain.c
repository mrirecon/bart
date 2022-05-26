/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */



#include <stddef.h>
#include <assert.h>

#include "misc/debug.h"

#include "num/ops.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"

#include "misc/misc.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/stack.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "chain.h"




struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b)
{
	assert(1 == nlop_get_nr_in_args(a));
	assert(1 == nlop_get_nr_out_args(a));
	assert(1 == nlop_get_nr_in_args(b));
	assert(1 == nlop_get_nr_out_args(b));

	const struct linop_s* la = linop_from_nlop(a);
	const struct linop_s* lb = linop_from_nlop(b);

	if ((NULL != la) && (NULL != lb)) {

		const struct linop_s* tmp = linop_chain(la, lb);
		linop_free(la);
		linop_free(lb);
		return nlop_from_linop_F(tmp);
	}


	PTR_ALLOC(struct nlop_s, n);

	const struct linop_s* (*der)[1][1] = TYPE_ALLOC(const struct linop_s*[1][1]);
	n->derivative = &(*der)[0][0];

	if (NULL == la)
		la = linop_clone(a->derivative[0]);

	if (NULL == lb)
		lb = linop_clone(b->derivative[0]);

	n->op = operator_chain(a->op, b->op);
	n->derivative[0] = linop_chain(la, lb);

	linop_free(la);
	linop_free(lb);

	return PTR_PASS(n);
}

struct nlop_s* nlop_chain_FF(const struct nlop_s* a, const struct nlop_s* b)
{
	struct nlop_s* x = nlop_chain(a, b);
	nlop_free(a);
	nlop_free(b);
	return x;
}


/**
 * Chain output o of nlop a in input i of nlop b.
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a
 * @param o
 * @param b
 * @param i
 */
struct nlop_s* nlop_chain2(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
//	int ai = nlop_get_nr_in_args(a);
//	int ao = nlop_get_nr_out_args(a);
//	int bi = nlop_get_nr_in_args(b);
	int bo = nlop_get_nr_out_args(b);
#if 0
	if ((1 == ai) && (1 == ao) && (1 == bi) && (1 == bo)) {

		assert((0 == o) && (0 == i));
		return nlop_chain(a, b);
	}
#endif

	auto domo = nlop_generic_codomain(a, o);
	auto domi = nlop_generic_domain(b, i);

	if (!iovec_check(domo, domi->N, domi->dims, domi->strs)) {

		nlop_debug(DP_INFO, a);
		nlop_debug(DP_INFO, b);
		error("Cannot chain args %d -> %d!\n", o, i);
	}

	struct nlop_s* nl = nlop_combine(b, a);
	struct nlop_s* li = nlop_link(nl, bo + o, i);
	nlop_free(nl);

	return li;
}


/**
 * Chain output o of nlop a in input i of nlop b.
 * Keep output o of a.
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o, a_o+1, ..., a_n]
 *
 * @param a
 * @param o
 * @param b
 * @param i
 */
struct nlop_s* nlop_chain2_keep(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
	auto iov = nlop_generic_domain(b, i);

	int Ob = nlop_get_nr_out_args(b);

	auto nb = nlop_from_linop_F(linop_identity_create(iov->N, iov->dims));
	nb = nlop_combine_FF(nb, nlop_clone(b));
	nb = nlop_dup_F(nb, 0, i + 1);

	auto result = nlop_chain2(a, o, nb, 0);
	nlop_free(nb);

	result = nlop_shift_output_F(result, Ob + o, 0);

	return result;
}


/**
 * Chain output o of nlop a in input i of nlop b.
 * Frees a and b.
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a
 * @param o
 * @param b
 * @param i
 */
struct nlop_s* nlop_chain2_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
	auto result = nlop_chain2(a, o, b, i);

	nlop_free(a);
	nlop_free(b);

	return result;
}


/**
 * Chain output o of nlop a in input i of nlop b.
 * Keep output o of a.
 * Frees a and b.
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, b_i+1, ..., b_n, a_0, ..., a_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o, a_o+1, ..., a_n]
 *
 * @param a
 * @param o
 * @param b
 * @param i
 */
struct nlop_s* nlop_chain2_keep_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
	auto result = nlop_chain2_keep(a, o, b, i);

	nlop_free(a);
	nlop_free(b);

	return result;
}


/**
 * Chain output o of nlop a in input i of nlop b.
 * Permutes inputs.
 * Frees a and b.
 *
 * Returned operator has
 * - inputs:  [a_0, ..., a_n, b_0, ..., b_i-1, b_i+1, ..., b_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o+1, ..., a_n]
 *
 * @param a
 * @param o
 * @param b
 * @param i
 */
struct nlop_s* nlop_chain2_swap_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
	auto result = nlop_chain2(a, o, b, i);

	int II = nlop_get_nr_in_args(result);
	int Ia = nlop_get_nr_in_args(a);
	int permute_array[II];

	for (int i = 0; i < II; i++)
		permute_array[(Ia + i) % II] = i;

	result = nlop_permute_inputs_F(result, II, permute_array);

	nlop_free(a);
	nlop_free(b);

	return result;
}

/**
 * Chain output o of nlop a in input i of nlop b.
 * Keep output o of a.
 * Permutes inputs.
 * Frees a and b.
 *
 * Returned operator has
 * - inputs:  [a_0, ..., a_n, b_0, ..., b_i-1, b_i+1, ..., b_n]
 * - outputs: [b_0, ..., b_n, a_0, ..., a_o-1, a_o, a_o+1, ..., a_n]
 *
 * @param a
 * @param o
 * @param b
 * @param i
 */
struct nlop_s* nlop_chain2_keep_swap_FF(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
	auto result = nlop_chain2_keep(a, o, b, i);

	int II = nlop_get_nr_in_args(result);
	int Ia = nlop_get_nr_in_args(a);
	int permute_array[II];

	for (int i = 0; i < II; i++)
		permute_array[(Ia + i) % II] = i;

	result = nlop_permute_inputs_F(result, II, permute_array);

	nlop_free(a);
	nlop_free(b);

	return result;
}


/**
 * Chain output o of nlop a into b and permute output to o
 *
 * Returned operator has
 * - inputs:  [a_0, ..., a_n]
 * - outputs: [a_0, ..., a_o-1, b_0, a_o+1, ..., a_n]
 *
 * @param a
 * @param o
 * @param b
 */
struct nlop_s* nlop_append_FF(const struct nlop_s* a, int o, const struct nlop_s* b)
{
	assert(1 == nlop_get_nr_in_args(b));
	assert(1 == nlop_get_nr_out_args(b));

	auto result = nlop_chain2_FF(a, o, b, 0);
	return nlop_shift_output_F(result, o, 0);
}


/**
 * Chain nlop a into input i of b and permute input of a to i
 *
 * Returned operator has
 * - inputs:  [b_0, ..., b_i-1, a_0, b_i+1, ..., b_n]
 * - outputs: [b_0, ..., b_n]
 *
 * @param a
 * @param b
 * @param i
 */
struct nlop_s* nlop_prepend_FF(const struct nlop_s* a, const struct nlop_s* b, int i)
{
	assert(1 == nlop_get_nr_in_args(a));
	assert(1 == nlop_get_nr_out_args(a));

	auto result = nlop_chain2_swap_FF(a, 0, b, i);

	return nlop_shift_input_F(result, i, 0);
}


/*
 * CAVE: if we pass the same operator twice, it might not
 * as they store state with respect to the derivative
 */
struct nlop_s* nlop_combine(const struct nlop_s* a, const struct nlop_s* b)
{
	assert(a != b);	// could also be deeply nested, but we do not detect it

	int ai = nlop_get_nr_in_args(a);
	int ao = nlop_get_nr_out_args(a);
	int bi = nlop_get_nr_in_args(b);
	int bo = nlop_get_nr_out_args(b);

	PTR_ALLOC(struct nlop_s, n);

	int II = ai + bi;
	int OO = ao + bo;

	auto der = TYPE_ALLOC(const struct linop_s*[II?:1][OO?:1]);
	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++) {
		for (int o = 0; o < OO; o++) {

			if ((i < ai) && (o < ao))
				(*der)[i][o] = linop_clone(nlop_get_derivative(a, o, i));
			else
			if ((ai <= i) && (ao <= o))
				(*der)[i][o] = linop_clone(nlop_get_derivative(b, o - ao, i - ai));
			else
			if ((i < ai) && (ao <= o)) {

				auto dom = nlop_generic_domain(a, i);
				auto cod = nlop_generic_codomain(b, o - ao);

				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(cod->N,
					cod->dims, cod->strs, dom->N, dom->dims, dom->strs);

			} else
			if ((ai <= i) && (o < ao)) {

				auto dom = nlop_generic_domain(b, i - ai);
				auto cod = nlop_generic_codomain(a, o);

				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(cod->N,
					cod->dims, cod->strs, dom->N, dom->dims, dom->strs);
			}
		}
	}


	auto cop = operator_combi_create(2, (const struct operator_s*[]){ a->op, b->op });

	assert(II == (int)operator_nr_in_args(cop));
	assert(OO == (int)operator_nr_out_args(cop));

	int perm[II + OO];	// ao ai bo bi -> ao bo ai bi
	int p = 0;

	for (int i = 0; i < ao; i++)
		perm[p++] = i;

	for (int i = 0; i < bo; i++)
		perm[p++] = (ao + ai + i);

	for (int i = 0; i < ai; i++)
		perm[p++] = (ao + i);

	for (int i = 0; i < bi; i++)
		perm[p++] = (ao + ai + bo + i);

	assert(II + OO == p);

	n->op = operator_permute(cop, II + OO, perm);
	operator_free(cop);

	return PTR_PASS(n);
}


struct nlop_s* nlop_combine_FF(const struct nlop_s* a, const struct nlop_s* b)
{
	auto result = nlop_combine(a, b);

	nlop_free(a);
	nlop_free(b);

	return result;
}



struct nlop_s* nlop_link(const struct nlop_s* x, int oo, int ii)
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(ii < II);
	assert(oo < OO);

	PTR_ALLOC(struct nlop_s, n);
	PTR_ALLOC(const struct linop_s*[II - 1][OO - 1], der);

	//assert(operator_ioflags(x->op) == ((1u << OO) - 1));

	n->op = operator_link_create(x->op, oo, OO + ii);

	//assert(operator_ioflags(n->op) == ((1u << (OO - 1)) - 1));

	// f(x_1, ..., g(x_n+1, ..., x_n+m), ..., xn)

	for (int i = 0, ip = 0; i < II - 1; i++, ip++) {

		if (i == ii)
			ip++;

		for (int o = 0, op = 0; o < OO - 1; o++, op++) {

			if (o == oo)
				op++;

			const struct linop_s* tmp = linop_chain(nlop_get_derivative(x, oo, ip),
								nlop_get_derivative(x, op, ii));

			(*der)[i][o] = linop_plus(nlop_get_derivative(x, op, ip), tmp);

			linop_free(tmp);
		}
	}

	n->derivative = &(*PTR_PASS(der))[0][0];

	return PTR_PASS(n);
}


struct nlop_s* nlop_link_F(const struct nlop_s* x, int oo, int ii)
{
	auto result = nlop_link(x, oo, ii);
	nlop_free(x);

	return result;
}


struct nlop_s* nlop_dup(const struct nlop_s* x, int a, int b)
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(a < II);
	assert(b < II);
	assert(a < b);

	auto doma = nlop_generic_domain(x, a);
	auto domb = nlop_generic_domain(x, b);

	if (!iovec_check(doma, domb->N, domb->dims, domb->strs)) {

		nlop_debug(DP_INFO, x);
		error("Cannot dup args %d and %d!\n", a, b);
	}

	PTR_ALLOC(struct nlop_s, n);
	PTR_ALLOC(const struct linop_s*[II-1][OO], der);

	//assert(operator_ioflags(x->op) == ((1u << OO) - 1));

	n->op = operator_dup_create(x->op, OO + a, OO + b);

	//assert(operator_ioflags(n->op) == ((1u << OO) - 1));

	// f(x_1, ..., xa, ... xa, ..., xn)

	for (int i = 0, ip = 0; i < II - 1; i++, ip++) {

		if (i == b)
			ip++;

		for (int o = 0; o < OO; o++) {

			if (i == a)
				(*der)[i][o] = linop_plus(nlop_get_derivative(x, o, ip), nlop_get_derivative(x, o, b));
			else
				(*der)[i][o] = linop_clone(nlop_get_derivative(x, o, ip));

		}
	}

	n->derivative = &(*PTR_PASS(der))[0][0];

	return PTR_PASS(n);
}

static const struct nlop_s* nlop_dup_generic(const struct nlop_s* x, int II, const int index[II])
{
	x = nlop_clone(x);
	for (int i = 1; i < II; i++)
		x = nlop_dup_F(x, index[0], index[i] + 1 -i);

	return x;
}

static struct nlop_s* nlop_stack_inputs_generic(const struct nlop_s* x, int NI, int _index[NI], int stack_dim)
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	int index[NI];
	for (int i = 0; i < NI; i++)
		index[i] = _index[i];

	int N = nlop_generic_domain(x, index[0])->N;

	if (0 > stack_dim)
		stack_dim += N;

	long odims[NI][N];
	long idims[N];
	md_copy_dims(N, idims, nlop_generic_domain(x, index[0])->dims);
	idims[stack_dim] = 0;

	int nindex = index[0];

	for (int i = 0; i < NI; i++) {

		assert(N == (int)nlop_generic_domain(x, index[i])->N);

		md_copy_dims(N, odims[i], nlop_generic_domain(x, index[i])->dims);
		idims[stack_dim] += odims[i][stack_dim];

		nindex = MIN(nindex, index[i]);
	}

	auto result = nlop_combine_FF(nlop_clone(x), nlop_destack_generic_create(NI, N, odims, idims, stack_dim));

	for (int i = 0; i < NI; i++) {

		result = nlop_link_F(result, OO, index[i]);

		for (int j = i + 1; j < NI; j++)
			if (index[j] > index[i])
				index[j]--;
	}

	return nlop_shift_input_F(result, nindex, II - NI);
}


struct nlop_s* nlop_stack_inputs(const struct nlop_s* x, int a, int b, int stack_dim)
{
	return nlop_stack_inputs_generic(x, 2, (int [2]) {a, b}, stack_dim);
}

struct nlop_s* nlop_stack_inputs_F(const struct nlop_s* x, int a, int b, int stack_dim)
{
	auto result = nlop_stack_inputs(x, a, b, stack_dim);
	nlop_free(x);
	return result;
}

static struct nlop_s* nlop_stack_outputs_generic(const struct nlop_s* x, int NO, int _index[NO], int stack_dim)
{
	int index[NO];

	for (int i = 0; i < NO; i++)
		index[i] = _index[i];

	int N = nlop_generic_codomain(x, index[0])->N;

	if (0 > stack_dim)
		stack_dim += N;

	long idims[NO][N];
	long odims[N];

	md_copy_dims(N, odims, nlop_generic_codomain(x, index[0])->dims);
	odims[stack_dim] = 0;

	int nindex = index[0];

	for (int i = 0; i < NO; i++) {

		assert(N == (int)nlop_generic_codomain(x, index[i])->N);

		md_copy_dims(N, idims[i], nlop_generic_codomain(x, index[i])->dims);
		odims[stack_dim] += idims[i][stack_dim];

		nindex = MIN(nindex, index[i]);
	}

	auto result = nlop_combine_FF(nlop_stack_generic_create(NO, N, odims, idims, stack_dim), nlop_clone(x));

	for (int i = 0; i < NO; i++) {

		result = nlop_link_F(result, 1 + index[i], 0);

		for (int j = i + 1; j < NO; j++)
			if (index[j] > index[i])
				index[j]--;
	}

	return nlop_shift_output_F(result, nindex, 0);
}

struct nlop_s* nlop_stack_outputs(const struct nlop_s* x, int a, int b, int stack_dim)
{
	return nlop_stack_outputs_generic(x, 2, (int [2]) {a, b}, stack_dim);
}

struct nlop_s* nlop_stack_outputs_F(const struct nlop_s* x, int a, int b, int stack_dim)
{
	auto result = nlop_stack_outputs(x, a, b, stack_dim);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_dup_F(const struct nlop_s* x, int a, int b)
{
	auto result = nlop_dup(x, a, b);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_permute_inputs(const struct nlop_s* x, int I2, const int perm[I2])
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(II == I2);

	PTR_ALLOC(struct nlop_s, n);

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*der)[i][o] = linop_clone(nlop_get_derivative(x, o, perm[i]));

	int perm2[II + OO];

	for (int i = 0; i < II + OO; i++)
		perm2[i] = (i < OO) ? i : (OO + perm[i - OO]);

	n->op = operator_permute(x->op, II + OO, perm2);

	return PTR_PASS(n);
}

struct nlop_s* nlop_permute_inputs_F(const struct nlop_s* x, int I2, const int perm[I2])
{
	auto result = nlop_permute_inputs(x, I2, perm);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_permute_outputs(const struct nlop_s* x, int O2, const int perm[O2])
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(OO == O2);

	PTR_ALLOC(struct nlop_s, n);

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
	n->derivative = &(*der)[0][0];

	for (int i = 0; i < II; i++)
		for (int o = 0; o < OO; o++)
			(*der)[i][o] = linop_clone(nlop_get_derivative(x, perm[o], i));


	int perm2[II + OO];

	for (int i = 0; i < II + OO; i++)
		perm2[i] = ((i < OO) ? perm[i] : i);

	n->op = operator_permute(x->op, II + OO, perm2);

	return PTR_PASS(n);
}

struct nlop_s* nlop_permute_outputs_F(const struct nlop_s* x, int O2, const int perm[O2])
{
	auto result = nlop_permute_outputs(x, O2, perm);
	nlop_free(x);
	return result;
}

struct nlop_s* nlop_shift_input(const struct nlop_s* x, int new_index, int old_index)
{
	int II = nlop_get_nr_in_args(x);
	assert(old_index < II);
	assert(new_index < II);

	int perm[II];
	for (int i = 0, ip = 0; i < II; i++, ip++) {

		perm[i] = ip;

		if (i == old_index)
			ip++;

		if (i == new_index)
			ip--;

		if (new_index > old_index)
			perm[i] = ip;
	}

	perm[new_index] = old_index;

	return nlop_permute_inputs(x, II, perm);
}

struct nlop_s* nlop_shift_input_F(const struct nlop_s* x, int new_index, int old_index)
{
	auto result = nlop_shift_input(x, new_index, old_index);
	nlop_free(x);

	return result;
}

struct nlop_s* nlop_shift_output(const struct nlop_s* x, int new_index, int old_index)
{
	int OO = nlop_get_nr_out_args(x);
	assert(old_index < OO);
	assert(new_index < OO);

	int perm[OO];

	for (int i = 0, ip = 0; i < OO; i++, ip++) {

		perm[i] = ip;

		if (i == old_index)
			ip++;

		if (i == new_index)
			ip--;

		if (new_index > old_index)
			perm[i] = ip;
	}

	perm[new_index] = old_index;

	return nlop_permute_outputs(x, OO, perm);
}

struct nlop_s* nlop_shift_output_F(const struct nlop_s* x, int new_index, int old_index)
{
	auto result = nlop_shift_output(x, new_index, old_index);
	nlop_free(x);
	return result;
}


struct nlop_s* nlop_stack_multiple_F(int N, const struct nlop_s* nlops[N], int II, int in_stack_dim[II], int OO, int out_stack_dim[OO])
{
	auto result = (struct nlop_s*)nlops[0];

	for (int i = 1; i < N; i++)
		result = nlop_combine_FF(result, nlops[i]);

	for (int i = 0; i < II; i++) {

		int index[N];
		index[0] = i;

		for (int j = 1; j < N; j++)
			index[j] = index[j - 1] + II - i;

		struct nlop_s* tmp = NULL;

		if (0 > in_stack_dim[i])
			tmp = (struct nlop_s*)nlop_dup_generic(result, N, index);
		else
			tmp = nlop_stack_inputs_generic(result, N, index, in_stack_dim[i]);

		nlop_free(result);
		result = tmp;
	}

	for (int i = 0; i < OO; i++) {

		int index[N];
		index[0] = i;

		for (int j = 1; j < N; j++)
			index[j] = index[j - 1] + OO - i;

		assert(0 <= out_stack_dim[i]);

		struct nlop_s* tmp = nlop_stack_outputs_generic(result, N, index, out_stack_dim[i]);
		nlop_free(result);
		result = tmp;
	}

	return result;
}

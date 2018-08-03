/* Copyright 2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */



#include <stddef.h>
#include <assert.h>

#include "num/ops.h"
#include "num/iovec.h"

#include "misc/misc.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"

#include "linops/linop.h"

#include "chain.h"




struct nlop_s* nlop_chain(const struct nlop_s* a, const struct nlop_s* b)
{
	assert(1 == nlop_get_nr_in_args(a));
	assert(1 == nlop_get_nr_out_args(a));
	assert(1 == nlop_get_nr_in_args(b));
	assert(1 == nlop_get_nr_out_args(b));

	const struct linop_s* la = linop_from_nlop(a);
	const struct linop_s* lb = linop_from_nlop(b);

	if ((NULL != la) && (NULL != lb))
		return nlop_from_linop(linop_chain(la, lb));

	PTR_ALLOC(struct nlop_s, n);

	const struct linop_s* (*der)[1][1] = TYPE_ALLOC(const struct linop_s*[1][1]);
	n->derivative = &(*der)[0][0];

	if (NULL == la)
		la = a->derivative[0];

	if (NULL == lb)
		lb = b->derivative[0];

	n->op = operator_chain(a->op, b->op);
	n->derivative[0] = linop_chain(la, lb);

	return PTR_PASS(n);
}

struct nlop_s* nlop_chain_FF(const struct nlop_s* a, const struct nlop_s* b)
{
	struct nlop_s* x = nlop_chain(a, b);
	nlop_free(a);
	nlop_free(b);
	return x;
}


struct nlop_s* nlop_chain2(const struct nlop_s* a, int o, const struct nlop_s* b, int i)
{
//	int ai = nlop_get_nr_in_args(a);
	int ao = nlop_get_nr_out_args(a);
//	int bi = nlop_get_nr_in_args(b);
//	int bo = nlop_get_nr_out_args(b);
#if 0
	if ((1 == ai) && (1 == ao) && (1 == bi) && (1 == bo)) {

		assert((0 == o) && (0 == i));
		return nlop_chain(a, b);
	}
#endif

	struct nlop_s* nl = nlop_combine(b, a);
	struct nlop_s* li = nlop_link(nl, ao + o, i);
	nlop_free(nl);

	return li;
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

	const struct linop_s* (*der)[II][OO] = TYPE_ALLOC(const struct linop_s*[II][OO]);
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

				//assert(dom->N == cod->N);
				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(dom->N,
					cod->dims, cod->strs, dom->dims, dom->strs);

			} else
			if ((ai <= i) && (o < ao)) {

				auto dom = nlop_generic_domain(b, i - ai);
				auto cod = nlop_generic_codomain(a, o);

				assert(dom->N == cod->N);
				assert(sizeof(complex float) == dom->size);
				assert(sizeof(complex float) == cod->size);

				(*der)[i][o] = linop_null_create2(dom->N,
					cod->dims, cod->strs, dom->dims, dom->strs);
			}
		}
	}


	n->op = operator_combi_create(2, (const struct operator_s*[]){ a->op, b->op });

	assert(II == (int)operator_nr_in_args(n->op));
	assert(OO == (int)operator_nr_out_args(n->op));

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

	n->op = operator_permute(n->op, II + OO, perm);

	return PTR_PASS(n);
}



struct nlop_s* nlop_link(const struct nlop_s* x, int oo, int ii)
{
	int II = nlop_get_nr_in_args(x);
	int OO = nlop_get_nr_out_args(x);

	assert(ii < II);
	assert(oo < OO);

	PTR_ALLOC(struct nlop_s, n);
	PTR_ALLOC(const struct linop_s*[II - 1][OO - 1], der);

	assert(operator_ioflags(x->op) == ((1u << OO) - 1));

	n->op = operator_link_create(x->op, oo, OO + ii);

	assert(operator_ioflags(n->op) == ((1u << (OO - 1)) - 1));

	// f(x_1, ..., g(x_n+1, ..., x_n+m), ..., xn)

	for (int i = 0, ip = 0; i < II - 1; i++, ip++) {

		if (i == ii)
			ip++;

		for (int o = 0, op = 0; o < OO - 1; o++, op++) {

			if (o == oo)
				op++;

			const struct linop_s* tmp = linop_chain(nlop_get_derivative(x, oo, ip),
								nlop_get_derivative(x, op, ii));

			(*der)[i][o] = linop_plus(
				nlop_get_derivative(x, op, ip),
				tmp);

			linop_free(tmp);
		}
	}

	n->derivative = &(*PTR_PASS(der))[0][0];

	return PTR_PASS(n);
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

	n->op = operator_permute(operator_ref(x->op), II + OO, perm2);

	return PTR_PASS(n);
}



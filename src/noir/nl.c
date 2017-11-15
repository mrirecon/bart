/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */


#include "num/multind.h"

#include "nlops/nlop.h"

#include "noir/model.h"

#include "misc/types.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "nl.h"

struct noir_op_s {

	INTERFACE(nlop_data_t);
	struct noir_data* data;
};

DEF_TYPEID(noir_op_s);

static void noir2_for(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct noir_op_s* data = CAST_DOWN(noir_op_s, _data);
	noir_fun(data->data, dst, src);
}

static void noir2_der(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct noir_op_s* data = CAST_DOWN(noir_op_s, _data);
	noir_der(data->data, dst, src);
}

static void noir2_adj(const nlop_data_t* _data, complex float* dst, const complex float* src)
{
	struct noir_op_s* data = CAST_DOWN(noir_op_s, _data);
	noir_adj(data->data, dst, src);
}

static void noir2_del(const nlop_data_t* _data)
{
	struct noir_op_s* data = CAST_DOWN(noir_op_s, _data);
	noir_free(data->data);
	xfree(data);
}

struct nlop_s* noir_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf)
{

	PTR_ALLOC(struct noir_op_s, data);
	SET_TYPEID(noir_op_s, data);

	data->data = noir_init(dims, mask, psf, conf);

	long idims[DIMS];
	md_select_dims(DIMS, conf->fft_flags|MAPS_FLAG|CSHIFT_FLAG, idims, dims);

	return nlop_create(DIMS, dims, DIMS, idims, CAST_UP(PTR_PASS(data)), noir2_for, noir2_der, noir2_adj, NULL, NULL, noir2_del);
}

struct noir_data* noir_get_data(struct nlop_s* op)
{
	struct noir_op_s* data = CAST_DOWN(noir_op_s, nlop_get_data(op));
	return data->data;
}


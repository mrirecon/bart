/* Copyright 2017. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */



struct noir_model_conf_s;
extern struct noir_model_conf_s noir_model_conf_defaults;

struct nlop_s;

extern struct nlop_s* noir_create(const long dims[DIMS], const complex float* mask, const complex float* psf, const struct noir_model_conf_s* conf);

struct noir_data;
extern struct noir_data* noir_get_data(struct nlop_s* op);

extern void nlop_free(const struct nlop_s*);


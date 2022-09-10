/* Copyright 2021-2022. Uecker Lab. University Medical Center Göttingen.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 **/

#include <complex.h>
#include <math.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"

#include "iter/iter.h"

#include "noncart/nufft.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/mri_ops.h"
#include "nlops/const.h"

#include "nn/data_list.h"

#include "misc.h"


struct network_data_s network_data_empty = {

	.ksp_dims = { 0 },
	.col_dims = { 0 },
	.psf_dims = { 0 },
	.img_dims = { 0 },
	.max_dims = { 0 },
	.cim_dims = { 0 },
	.out_dims = { 0 },
	.pat_dims = { 0 },
	.trj_dims = { 0 },
	.bas_dims = { 0 },
	.scl_dims = { 0 },

	.filename_trajectory = NULL,
	.filename_pattern = NULL,
	.filename_kspace = NULL,
	.filename_coil = NULL,
	.filename_out = NULL,
	.filename_basis = NULL,

	.filename_adjoint = NULL,
	.filename_psf = NULL,
	.export = false,

	.kspace = NULL,
	.coil = NULL,
	.psf = NULL,
	.pattern = NULL,
	.adjoint = NULL,
	.initialization = NULL,
	.out = NULL,
	.trajectory = NULL,
	.basis = NULL,
	.scale = NULL,

	.create_out = false,
	.load_mem = false,
	.batch_flags = SLICE_FLAG | AVG_FLAG | BATCH_FLAG,

	.nufft_conf = &nufft_conf_defaults,
};

static void load_mem(struct network_data_s* nd)
{
	if (!nd->load_mem)
		return;

	complex float* tmp_psf = anon_cfl("", nd->ND, nd->psf_dims);
	complex float* tmp_adj = anon_cfl("", nd->N, nd->img_dims);
	complex float* tmp_col = anon_cfl("", nd->N, nd->col_dims);

	md_copy(nd->ND, nd->psf_dims, tmp_psf, nd->psf, CFL_SIZE);
	md_copy(nd->N, nd->img_dims, tmp_adj, nd->adjoint, CFL_SIZE);
	md_copy(nd->N, nd->col_dims, tmp_col, nd->coil, CFL_SIZE);

	unmap_cfl(nd->ND, nd->psf_dims, nd->psf);
	unmap_cfl(DIMS, nd->img_dims, nd->adjoint);
	unmap_cfl(DIMS, nd->col_dims, nd->coil);

	nd->psf = tmp_psf;
	nd->adjoint = tmp_adj;
	nd->coil = tmp_col;

	if (!nd->create_out) {

		complex float* tmp_out = anon_cfl("", nd->N, nd->out_dims);
		md_copy(nd->N, nd->out_dims, tmp_out, nd->out, CFL_SIZE);
		unmap_cfl(DIMS, nd->out_dims, nd->out);
		nd->out = tmp_out;
	}
}

static void load_network_data_precomputed(struct network_data_s* nd)
{
	nd->N = DIMS;
	nd->ND = DIMS;

	assert(NULL != nd->filename_adjoint);
	assert(NULL != nd->filename_psf);

	nd->coil = load_cfl(nd->filename_coil, DIMS, nd->col_dims);
	nd->adjoint = load_cfl(nd->filename_adjoint, DIMS, nd->img_dims);
	nd->psf = load_cfl(nd->filename_psf, DIMS + 1, nd->psf_dims);
	nd->kspace = load_cfl(nd->filename_kspace, DIMS, nd->ksp_dims);
	unmap_cfl(DIMS, nd->ksp_dims, nd->kspace);
	nd->kspace = NULL;

	nd->ND = (1 != nd->psf_dims[DIMS]) ? DIMS + 1 : DIMS;

	md_max_dims(nd->N, ~0, nd->max_dims, nd->img_dims, nd->col_dims);
	md_select_dims(nd->N, ~MAPS_FLAG, nd->cim_dims, nd->max_dims);

	if (nd->filename_basis != NULL) {

		nd->basis = load_cfl(nd->filename_basis, DIMS, nd->bas_dims);
		assert(nd->psf_dims[5] == nd->psf_dims[6]);
	}

	if (nd->create_out) {

		md_copy_dims(DIMS, nd->out_dims, nd->img_dims);
		nd->out = create_cfl(nd->filename_out, DIMS, nd->img_dims);
	} else {

		nd->out = load_cfl(nd->filename_out, DIMS, nd->out_dims);
		assert(    md_check_equal_dims(DIMS, nd->img_dims, nd->out_dims, ~0)
			|| md_check_equal_dims(DIMS, nd->cim_dims, nd->out_dims, ~0) );
	}
}

static void compute_adjoint_cart(struct network_data_s* nd)
{
	assert(NULL == nd->filename_basis);
	nd->adjoint = (nd->export) ? create_cfl(nd->filename_adjoint, DIMS, nd->img_dims) : anon_cfl("", DIMS, nd->img_dims);

	long ksp_dims_s[DIMS];
	long img_dims_s[DIMS];
	long col_dims_s[DIMS];
	long pat_dims_s[DIMS];

	md_select_dims(DIMS, ~nd->batch_flags, ksp_dims_s, nd->ksp_dims);
	md_select_dims(DIMS, ~nd->batch_flags, img_dims_s, nd->img_dims);
	md_select_dims(DIMS, ~nd->batch_flags, col_dims_s, nd->col_dims);
	md_select_dims(DIMS, ~nd->batch_flags, pat_dims_s, nd->pat_dims);

	auto model = sense_cart_create(nd->N, ksp_dims_s, img_dims_s, col_dims_s, pat_dims_s);
	auto sense_adjoint = nlop_sense_adjoint_create(1, &model, false);

	int DO[1] = { DIMS };
	int DI[3] = { DIMS, DIMS, DIMS };

	const long* odims[1] = { nd->img_dims };
	const long* idims[3] = { nd->ksp_dims, nd->col_dims, nd->pat_dims};

	complex float* dst[1] = { nd->adjoint };
	const complex float* src[3] = { nd->kspace, nd->coil, nd->pattern };

	nlop_generic_apply_loop_sameplace(sense_adjoint, nd->batch_flags, 1, DO, odims, dst, 3, DI, idims, src, nd->adjoint);

	nlop_free(sense_adjoint);
	sense_model_free(model);

	md_copy_dims(DIMS, nd->psf_dims, nd->pat_dims);
	nd->psf = (nd->export) ? create_cfl(nd->filename_psf, DIMS, nd->psf_dims) : anon_cfl("", DIMS, nd->psf_dims);
	md_copy(DIMS, nd->pat_dims, nd->psf, nd->pattern, CFL_SIZE);
}

static void compute_adjoint_noncart(struct network_data_s* nd)
{
	nd->trajectory = load_cfl(nd->filename_trajectory, DIMS, nd->trj_dims);

	if (NULL != nd->filename_basis) {

		nd->basis = load_cfl(nd->filename_basis, DIMS, nd->bas_dims);

		md_copy_dims(DIMS, nd->max_dims, nd->ksp_dims);
		md_copy_dims(5, nd->max_dims, nd->col_dims);
		md_max_dims(DIMS, ~0, nd->max_dims, nd->max_dims, nd->bas_dims);
		nd->max_dims[TE_DIM] = 1;

		md_select_dims(DIMS, ~MAPS_FLAG, nd->cim_dims, nd->max_dims);
		md_select_dims(DIMS, ~COIL_FLAG, nd->img_dims, nd->max_dims);

	} else {

		md_singleton_dims(DIMS, nd->bas_dims);
	}

	nd->adjoint = (nd->export) ? create_cfl(nd->filename_adjoint, DIMS, nd->img_dims) : anon_cfl("", DIMS, nd->img_dims);

	long max_dims_s[DIMS];
	long ksp_dims_s[DIMS];
	long img_dims_s[DIMS];
	long cim_dims_s[DIMS];
	long trj_dims_s[DIMS];
	long pat_dims_s[DIMS];
	long col_dims_s[DIMS];

	md_select_dims(DIMS, ~nd->batch_flags, max_dims_s, nd->max_dims);
	md_select_dims(DIMS, ~nd->batch_flags, ksp_dims_s, nd->ksp_dims);
	md_select_dims(DIMS, ~nd->batch_flags, img_dims_s, nd->img_dims);
	md_select_dims(DIMS, ~nd->batch_flags, cim_dims_s, nd->cim_dims);
	md_select_dims(DIMS, ~nd->batch_flags, trj_dims_s, nd->trj_dims);
	md_select_dims(DIMS, ~nd->batch_flags, pat_dims_s, nd->pat_dims);
	md_select_dims(DIMS, ~nd->batch_flags, col_dims_s, nd->col_dims);

	auto model = sense_noncart_create(nd->N, trj_dims_s, pat_dims_s, ksp_dims_s, cim_dims_s, img_dims_s, col_dims_s, nd->bas_dims, nd->basis, *(nd->nufft_conf));
	auto sense_adjoint = nlop_sense_adjoint_create(1, &model, true);

	nd->ND = DIMS + 1;
	md_copy_dims(DIMS + 1, nd->psf_dims, nlop_generic_codomain(sense_adjoint, 1)->dims);
	nd->psf_dims[BATCH_DIM] = nd->max_dims[BATCH_DIM];
	nd->psf = (nd->export) ? create_cfl(nd->filename_psf, DIMS + 1, nd->psf_dims) : anon_cfl("", DIMS + 1, nd->psf_dims);

	int DO[2] = { DIMS, DIMS + 1 };
	int DI[4] = { DIMS, DIMS, DIMS, DIMS };

	const long* odims[2] = { nd->img_dims, nd->psf_dims };
	const long* idims[4] = { nd->ksp_dims, nd->col_dims, nd->pat_dims, nd->trj_dims };

	complex float* dst[2] = { nd->adjoint, nd->psf };
	const complex float* src[4] = { nd->kspace, nd->coil, nd->pattern, nd->trajectory };

	nlop_generic_apply_loop_sameplace(sense_adjoint, nd->batch_flags, 2, DO, odims, dst, 4, DI, idims, src, nd->adjoint);


	nlop_free(sense_adjoint);
	sense_model_free(model);
}

void load_network_data(struct network_data_s* nd) {

	nd->N = DIMS;
	nd->ND = DIMS;

	if ( !nd->export && (NULL != nd->filename_adjoint)) {

		load_network_data_precomputed(nd);
		load_mem(nd);
		return;
	}

	nd->coil = load_cfl(nd->filename_coil, DIMS, nd->col_dims);
	nd->kspace = load_cfl(nd->filename_kspace, DIMS, nd->ksp_dims);

	if (NULL != nd->filename_pattern) {

		nd->pattern = load_cfl(nd->filename_pattern, DIMS, nd->pat_dims);
	} else {

		md_select_dims(DIMS, ~(COIL_FLAG), nd->pat_dims, nd->ksp_dims);
		nd->pattern = anon_cfl("", DIMS, nd->pat_dims);
		estimate_pattern(DIMS, nd->ksp_dims, COIL_FLAG, nd->pattern, nd->kspace);
	}


	//remove const dims in pattern
	long pat_dims[DIMS];
	long pat_strs[DIMS];
	md_copy_dims(DIMS, pat_dims, nd->pat_dims);
	md_calc_strides(DIMS, pat_strs, nd->pat_dims, CFL_SIZE);

	for (int i = 0; i < DIMS; i++) {

		long pat_strs2[DIMS];
		md_copy_dims(DIMS, pat_strs2, pat_strs);
		pat_strs2[i] = 0;

		if ( (1 == pat_dims[i]) || md_compare2(DIMS, pat_dims, pat_strs2, nd->pattern, pat_strs, nd->pattern, CFL_SIZE) )
			pat_dims[i] = 1;
	}

	if (!md_check_equal_dims(DIMS, pat_dims, nd->pat_dims, ~0)) {

		complex float* tmp = anon_cfl("", DIMS, pat_dims);
		md_copy2(DIMS, pat_dims, MD_STRIDES(DIMS, pat_dims, CFL_SIZE), tmp, pat_strs, nd->pattern, CFL_SIZE);
		unmap_cfl(DIMS, nd->pat_dims, nd->pattern);

		nd->pattern = tmp;
		md_copy_dims(DIMS, nd->pat_dims, pat_dims);
	}



	//remove frequency oversampling
	if (!md_check_equal_dims(DIMS, nd->ksp_dims, nd->col_dims, FFT_FLAGS & (~md_nontriv_dims(DIMS, nd->pat_dims))) && (NULL == nd->filename_trajectory)) {

		long ksp_dims[DIMS];
		md_copy_dims(DIMS, ksp_dims, nd->ksp_dims);

		for (int i = 0; i < DIMS; i++)
			if ((nd->ksp_dims[i] != nd->col_dims[i]) && (1 == nd->pat_dims[i]) && MD_IS_SET(FFT_FLAGS, i))
				ksp_dims[i] = nd->col_dims[i];

		complex float* tmp = anon_cfl("", DIMS, ksp_dims);

		ifftuc(DIMS, nd->ksp_dims, FFT_FLAGS, nd->kspace, nd->kspace);
		md_resize_center(DIMS, ksp_dims, tmp, nd->ksp_dims, nd->kspace, CFL_SIZE);
		fftuc(DIMS, ksp_dims, FFT_FLAGS, tmp, tmp);

		unmap_cfl(DIMS, nd->ksp_dims, nd->kspace);
		nd->kspace = tmp;

		md_copy_dims(DIMS, nd->ksp_dims, ksp_dims);
	}

	md_copy_dims(DIMS, nd->max_dims, nd->ksp_dims);
	md_copy_dims(5, nd->max_dims, nd->col_dims);

	md_select_dims(DIMS, ~MAPS_FLAG, nd->cim_dims, nd->max_dims);
	md_select_dims(DIMS, ~COIL_FLAG, nd->img_dims, nd->max_dims);

	if (NULL != nd->filename_trajectory)
		compute_adjoint_noncart(nd);
	else
		compute_adjoint_cart(nd);

	unmap_cfl(DIMS, nd->pat_dims, nd->pattern);
	md_singleton_dims(DIMS, nd->pat_dims);
	nd->pattern = NULL;

	if (nd->create_out) {

		md_copy_dims(DIMS, nd->out_dims, nd->img_dims);
		nd->out = create_cfl(nd->filename_out, DIMS, nd->img_dims);
	} else {

		nd->out = load_cfl(nd->filename_out, DIMS, nd->out_dims);
		assert(    md_check_equal_dims(DIMS, nd->img_dims, nd->out_dims, ~0)
			|| md_check_equal_dims(DIMS, nd->cim_dims, nd->out_dims, ~0) );
	}

	load_mem(nd);
}


void network_data_compute_init(struct network_data_s* nd, complex float lambda, int cg_iter)
{
	assert(NULL == nd->initialization);
	nd->initialization = anon_cfl("", nd->N, nd->img_dims);

	int N = nd->N;
	int ND = nd->ND;

	long max_dims2[N];
	long psf_dims2[ND];

	unsigned long loop_flags = nd->batch_flags;

	md_select_dims(N, ~loop_flags, max_dims2, nd->max_dims);
	md_select_dims(ND, ~loop_flags, psf_dims2, nd->psf_dims);

	struct config_nlop_mri_s conf2 = conf_nlop_mri_simple;
	conf2.pattern_flags = md_nontriv_dims(ND, psf_dims2);
	conf2.noncart = nd->N != nd->ND;
	conf2.nufft_conf = nd->nufft_conf;

	if (NULL != nd->basis)
		conf2.basis_flags = COEFF_FLAG | TE_FLAG;

	struct iter_conjgrad_conf iter_conf = iter_conjgrad_defaults;
	iter_conf.l2lambda = 0;
	iter_conf.maxiter = cg_iter;

	auto nlop_normal_inv = nlop_mri_normal_inv_create(N, max_dims2, MD_SINGLETON_DIMS(N), ND, psf_dims2, &conf2, &iter_conf);
	nlop_normal_inv = nlop_set_input_const_F(nlop_normal_inv, 3, N, MD_SINGLETON_DIMS(N), true, &lambda);

	int DO[1] = { nd->N };
	int DI[3] = { nd->N, nd->N, nd->ND };

	const long* odims[1] = { nd->img_dims };
	const long* idims[3] = { nd->img_dims, nd->col_dims, nd->psf_dims};

	complex float* dst[1] = { nd->initialization };
	const complex float* src[3] = { nd->adjoint, nd->coil, nd->psf };

	nlop_generic_apply_loop_sameplace(nlop_normal_inv, loop_flags, 1, DO, odims, dst, 3, DI, idims, src, nd->adjoint);

	nlop_free(nlop_normal_inv);
}

void network_data_normalize(struct network_data_s* nd)
{
	int N = nd->N;

	long sstrs[N];
	long istrs[N];

	md_select_dims(N, nd->batch_flags & ~SLICE_FLAG, nd->scl_dims, nd->img_dims);
	md_calc_strides(N, sstrs, nd->scl_dims, CFL_SIZE);
	md_calc_strides(N, istrs, nd->img_dims, CFL_SIZE);

	complex float* tmp = md_alloc(N, nd->img_dims, CFL_SIZE);
	md_zabs(N, nd->img_dims, tmp, nd->initialization ? nd->initialization : nd->adjoint);

	assert(NULL == nd->scale);
	nd->scale = anon_cfl("", N, nd->scl_dims);
	md_clear(N, nd->scl_dims, nd->scale, CFL_SIZE);
	md_zmax2(N, nd->img_dims, sstrs, nd->scale, sstrs, nd->scale, istrs, tmp);
	md_free(tmp);

	md_zspow(N, nd->scl_dims, nd->scale, nd->scale, -1);
}



void free_network_data(struct network_data_s* nd)
{
	unmap_cfl(nd->ND, nd->psf_dims, nd->psf);
	unmap_cfl(DIMS, nd->img_dims, nd->adjoint);
	unmap_cfl(DIMS, nd->col_dims, nd->coil);
	unmap_cfl(DIMS, nd->img_dims, nd->out);
	if(NULL != nd->trajectory)
		unmap_cfl(DIMS, nd->trj_dims, nd->trajectory);
	if(NULL != nd->pattern)
		unmap_cfl(DIMS, nd->pat_dims, nd->pattern);
	if(NULL != nd->basis)
		unmap_cfl(DIMS, nd->bas_dims, nd->basis);
	if(NULL != nd->scale)
		unmap_cfl(DIMS, nd->scl_dims, nd->scale);
	if(NULL != nd->initialization)
		unmap_cfl(DIMS, nd->img_dims, nd->initialization);
}

//move all batch dimensions to the last one, unfold dimensions if necessary
static void merge_slice_to_batch_dim(int N, const long bat_dims[N], long dims[N], complex float** data)
{
	unsigned long bat_flags = md_nontriv_dims(N, bat_dims);
	
	long tbat_dims[N];
	long tdims[N];

	md_select_dims(N, bat_flags, tbat_dims, dims);
	md_copy_dims(N, tdims, dims);

	if ((NULL != data) && (NULL != *data) && (0 != md_nontriv_dims(N, tbat_dims)) && !md_check_equal_dims(N, tbat_dims, bat_dims, ~0)){

		for (int i = 0; i < N; i++)
			if (MD_IS_SET(bat_flags, i))
				tdims[i] = bat_dims[i];
		
		complex float* tdata = anon_cfl("", N, tdims);
		md_copy2(N, tdims, MD_STRIDES(N, tdims, CFL_SIZE), tdata, MD_STRIDES(N, dims, CFL_SIZE), *data, CFL_SIZE);
		unmap_cfl(N, dims, *data);
		*data = tdata;
	}

	md_select_dims(N, bat_flags, tbat_dims, tdims);
	md_select_dims(N, ~bat_flags, dims, tdims);

	dims[BATCH_DIM] = md_calc_size(N, tbat_dims);
}

void network_data_slice_dim_to_batch_dim(struct network_data_s* nd)
{
	long bat_dims[nd->ND];
	md_singleton_dims(nd->ND, bat_dims);
	md_select_dims(nd->N, nd->batch_flags, bat_dims, nd->img_dims);

	long img_dims[nd->N];
	md_copy_dims(nd->N, img_dims, nd->img_dims);

	merge_slice_to_batch_dim(nd->N, bat_dims, nd->scl_dims, &(nd->scale));
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->ksp_dims, &(nd->kspace));
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->col_dims, &(nd->coil));
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->psf_dims, &(nd->psf));
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->img_dims, &(nd->adjoint));
	merge_slice_to_batch_dim(nd->N, bat_dims, img_dims, &(nd->initialization));
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->max_dims, NULL);
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->cim_dims, NULL);
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->out_dims, &(nd->out));
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->pat_dims, &(nd->pattern));
	merge_slice_to_batch_dim(nd->N, bat_dims, nd->trj_dims, &(nd->trajectory));

}

struct named_data_list_s* network_data_get_named_list(struct network_data_s* nd)
{
	auto train_data_list = named_data_list_create();
	
	if (NULL != nd->kspace)
		named_data_list_append(train_data_list, nd->N, nd->ksp_dims, nd->kspace, "kspace");
	
	if (NULL != nd->coil)
		named_data_list_append(train_data_list, nd->N, nd->col_dims, nd->coil, "coil");
	
	if (NULL != nd->psf)
		named_data_list_append(train_data_list, nd->ND, nd->psf_dims, nd->psf, "psf");
	
	if (NULL != nd->pattern)
		named_data_list_append(train_data_list, nd->N, nd->pat_dims, nd->pattern, "pattern");
		
	if (NULL != nd->adjoint)
		named_data_list_append(train_data_list, nd->N, nd->img_dims, nd->adjoint, "adjoint");

	if (NULL != nd->initialization)
		named_data_list_append(train_data_list, nd->N, nd->img_dims, nd->initialization, "initialization");

	if (NULL != nd->out)
		named_data_list_append(train_data_list, nd->N, nd->out_dims, nd->out, nd->create_out ? "reconstruction" : "reference");

	if (NULL != nd->trajectory)
		named_data_list_append(train_data_list, nd->N, nd->trj_dims, nd->trajectory, "trajectory");

	if (NULL != nd->basis)
		named_data_list_append(train_data_list, nd->N, nd->bas_dims, nd->basis, "basis");

	if (NULL != nd->scale)
		named_data_list_append(train_data_list, nd->N, nd->scl_dims, nd->scale, "scale");

	return train_data_list;
}

long network_data_get_tot(struct network_data_s* nd)
{
	long bat_dims[nd->N];
	md_select_dims(nd->N, nd->batch_flags, bat_dims, nd->img_dims);
	return md_calc_size(nd->N, bat_dims);
}

#include <complex.h>
#include <math.h>

#include "linops/fmac.h"
#include "linops/someops.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mmio.h"

#include "misc/mri.h"
#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/iovec.h"

#include "noncart/nufft.h"

#include "linops/linop.h"
#include "linops/fmac.h"

#include "nlops/nlop.h"
#include "nlops/mri_ops.h"

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
	.out = NULL,
	.trajectory = NULL,
	.basis = NULL,

	.create_out = false,
	.load_mem = false,

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

	md_select_dims(DIMS, ~BATCH_FLAG, ksp_dims_s, nd->ksp_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, img_dims_s, nd->img_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, col_dims_s, nd->col_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, pat_dims_s, nd->pat_dims);

	auto model = sense_cart_create(nd->N, ksp_dims_s, img_dims_s, col_dims_s, pat_dims_s);
	auto sense_adjoint = nlop_sense_adjoint_create(1, &model, false);

	int DO[1] = { DIMS };
	int DI[3] = { DIMS, DIMS, DIMS };

	const long* odims[1] = { nd->img_dims };
	const long* idims[3] = { nd->ksp_dims, nd->col_dims, nd->pat_dims};

	complex float* dst[1] = { nd->adjoint };
	const complex float* src[3] = { nd->kspace, nd->coil, nd->pattern };

	nlop_generic_apply_loop_sameplace(sense_adjoint, BATCH_FLAG, 1, DO, odims, dst, 3, DI, idims, src, nd->adjoint);

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

	md_select_dims(DIMS, ~BATCH_FLAG, max_dims_s, nd->max_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, ksp_dims_s, nd->ksp_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, img_dims_s, nd->img_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, cim_dims_s, nd->cim_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, trj_dims_s, nd->trj_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, pat_dims_s, nd->pat_dims);
	md_select_dims(DIMS, ~BATCH_FLAG, col_dims_s, nd->col_dims);

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

	nlop_generic_apply_loop_sameplace(sense_adjoint, BATCH_FLAG, 2, DO, odims, dst, 4, DI, idims, src, nd->adjoint);


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
}
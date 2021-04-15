#include <math.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/ops.h"
#include "num/rand.h"
#include "num/fft.h"

#include "iter/misc.h"
#include "iter/iter.h"
#include "iter/iter2.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/someops.h"

#include "nlops/nlop.h"
#include "nlops/cast.h"
#include "nlops/const.h"
#include "nlops/chain.h"
#include "nlops/tenmul.h"
#include "nlops/someops.h"

#include "mri_ops.h"


struct config_nlop_mri_s conf_nlop_mri_simple = {

	.coil_flags = ~(0ul),
	.image_flags = ~COIL_FLAG,
	.pattern_flags = FFT_FLAGS,
	.batch_flags = MD_BIT(4),
	.fft_flags = FFT_FLAGS,

	.gridded = false,
};

static bool test_idims_compatible(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	long tdims[N];
	md_select_dims(N, conf->image_flags, tdims, dims);
	return md_check_equal_dims(N, tdims, idims, ~(conf->fft_flags));
}








/**
 * Returns: MRI forward operator (SENSE Operator)
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 *
 * Output tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_forward_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	long cdims[N];
	long pdims[N];

	md_select_dims(N, conf->coil_flags, cdims, dims);
	md_select_dims(N, conf->pattern_flags, pdims, dims);

	for (int i = 0; i < N; i++)
		cdims[i] = MD_IS_SET(conf->fft_flags & conf->coil_flags, i) ? idims[i] : cdims[i];

	const struct nlop_s* result = nlop_tenmul_create(N, cdims, idims, cdims); //in: image, coil

	const struct linop_s* lop = linop_fftc_create(N, dims, conf->fft_flags);
	if (!md_check_equal_dims(N, cdims, dims, ~0))
		lop = linop_chain_FF(linop_resize_center_create(N, dims, cdims), lop);
	result = nlop_chain2_FF(result, 0, nlop_from_linop_F(lop), 0); //in: image, coil

	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, dims, dims, pdims), 0); //in: image, coil, pattern

	debug_printf(DP_DEBUG2, "mri forward created\n");
	return result;
}








/**
 * Returns: Adjoint MRI operator (SENSE Operator)
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1)
 *
 * Output tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1, Nb)
 */
const struct nlop_s* nlop_mri_adjoint_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{

	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	long cdims[N];
	long pdims[N];

	md_select_dims(N, conf->coil_flags, cdims, dims);
	md_select_dims(N, conf->pattern_flags, pdims, dims);

	for (int i = 0; i < N; i++)
		cdims[i] = MD_IS_SET(conf->fft_flags & conf->coil_flags, i) ? idims[i] : cdims[i];

	const struct linop_s* lop = linop_ifftc_create(N, dims, conf->fft_flags);
	if (!md_check_equal_dims(N, cdims, dims, ~0))
		lop = linop_chain_FF(lop, linop_resize_center_create(N, cdims, dims));

	const struct nlop_s* result = nlop_from_linop_F(lop);

	if (!conf->gridded)
		result = nlop_chain2_FF(nlop_tenmul_create(N, dims, dims, pdims), 0, result, 0); //in: kspace, pattern
	else
		result = nlop_combine_FF(result, nlop_del_out_create(N, pdims));

	result = nlop_chain2_swap_FF(result, 0, nlop_tenmul_create(N, idims, cdims, cdims), 0); //in: kspace, pattern, coil
	result = nlop_chain2_FF(nlop_from_linop_F(linop_zconj_create(N, cdims)), 0, result, 2); //in: kspace, pattern, coil
	result = nlop_shift_input_F(result, 1, 2); //in: kspace, coil, pattern

	debug_printf(DP_DEBUG2, "mri adjoint created\n");

	return result;
}








// this struct and the operators in it can be used to apply the normal eq operators and compute the regeularized inverse
struct mri_normal_operator_s {

	int N;

	const long* idims;
	const long* pdims;
	const long* cdims;
	const long* kdims;

	const long* bdims;

	const struct linop_s* lop_fft; //reusable
	const struct linop_s* lop_fft_mod; //reusable

	const struct operator_s** normal_ops;
};

static const struct operator_s* create_mri_normal_operator_int(	int N, const long kdims[N], const long cdims[N], const long idims[N], const long pdims[N],
								const complex float* coil, const complex float* pattern,
								const struct linop_s* lop_fft_mod, //reusable
								const struct linop_s* lop_fft //reusable
							)
{
	unsigned long image_flags = md_nontriv_dims(N, idims);
	unsigned long pattern_flags = md_nontriv_dims(N, pdims);

	auto linop_frw = linop_chain_FF(linop_clone(lop_fft_mod), linop_fmac_create(N, cdims, 0, ~(image_flags), 0, coil));

	if (!md_check_equal_dims(N, cdims, kdims, ~0))
		linop_frw = linop_chain_FF(linop_frw, linop_resize_center_create(N, kdims, cdims));

	linop_frw = linop_chain_FF(linop_frw, linop_clone(lop_fft));

	auto linop_pattern = linop_cdiag_create(N, kdims, pattern_flags, pattern);

	auto result = operator_chainN(3, (const struct operator_s **)MAKE_ARRAY(linop_frw->forward, linop_pattern->forward, linop_frw->adjoint));

	linop_free(linop_frw);
	linop_free(linop_pattern);

	return result;
}

static void create_mri_normal_operator(struct mri_normal_operator_s* d, const complex float* coil, const complex float* pattern)
{
	unsigned long batch_flags = md_nontriv_dims(d->N, d->bdims);

	long pdims_normal[d->N];
	long cdims_normal[d->N];
	long kdims_normal[d->N];
	long idims_normal[d->N];

	md_select_dims(d->N, ~(batch_flags), pdims_normal, d->pdims);
	md_select_dims(d->N, ~(batch_flags), cdims_normal, d->cdims);
	md_select_dims(d->N, ~(batch_flags), kdims_normal, d->kdims);
	md_select_dims(d->N, ~(batch_flags), idims_normal, d->idims);

	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;

	do {
		int index = md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos);
		if (NULL != (d->normal_ops)[index])
			operator_free((d->normal_ops)[index]);

		const complex float* coil_i = &MD_ACCESS(d->N, MD_STRIDES(d->N, d->cdims, CFL_SIZE), pos, coil);
		const complex float* pattern_i = &MD_ACCESS(d->N, MD_STRIDES(d->N, d->pdims, CFL_SIZE), pos, pattern);

		(d->normal_ops)[index] = create_mri_normal_operator_int(	d->N, kdims_normal, cdims_normal, idims_normal, pdims_normal,
										coil_i, pattern_i,
										d->lop_fft_mod,
										d->lop_fft
									);

	} while (md_next(d->N, d->bdims, ~(0ul), pos));
}

static struct mri_normal_operator_s* mri_normal_operator_data_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	PTR_ALLOC(struct mri_normal_operator_s, data);

	// batch dims must be outer most dims
	bool batch = false;
	for (int i = 0; i < N; i++) {

		if (MD_IS_SET(conf->batch_flags, i))
			batch = true;
		else
			assert(!batch || (1 == dims[i]));
	}

	data->N = N;

	PTR_ALLOC(long[N], nidims);
	PTR_ALLOC(long[N], cdims);
	PTR_ALLOC(long[N], kdims);
	PTR_ALLOC(long[N], pdims);
	PTR_ALLOC(long[N], bdims);

	md_copy_dims(N, *nidims, idims);
	md_copy_dims(N, *kdims, dims);
	md_select_dims(N, conf->coil_flags, *cdims, dims);
	md_select_dims(N, conf->pattern_flags, *pdims, dims);
	md_select_dims(N, conf->batch_flags, *bdims, dims);

	for (int i = 0; i < N; i++)
		(*cdims)[i] = MD_IS_SET(conf->fft_flags & conf->coil_flags, i) ? idims[i] : (*cdims)[i];

	data->idims = *PTR_PASS(nidims);
	data->pdims = *PTR_PASS(pdims);
	data->cdims = *PTR_PASS(cdims);
	data->kdims = *PTR_PASS(kdims);
	data->bdims = *PTR_PASS(bdims);

	PTR_ALLOC(const struct operator_s*[md_calc_size(N, data->bdims)], normalops);
	for (int i = 0; i < md_calc_size(N, data->bdims); i++)
		(*normalops)[i] = NULL;

	data->normal_ops = *PTR_PASS(normalops);

	long kdims_normal[N];
	md_select_dims(N, ~(conf->batch_flags), kdims_normal, data->kdims);
	data->lop_fft = linop_fft_create(N, kdims_normal, conf->fft_flags);

	// create linop for fftmod which only applies on coil dims not kdims
	long fft_dims[N];
	md_select_dims(N, conf->fft_flags, fft_dims, data->kdims);

	long fft_idims[N];
	md_select_dims(N, conf->fft_flags, fft_idims, idims);

	complex float* fftmod_k = md_alloc(N, fft_dims, CFL_SIZE);
	md_zfill(N, fft_dims, fftmod_k, 1.);
	fftmod(N, fft_dims, conf->fft_flags, fftmod_k, fftmod_k);
	fftscale(N, fft_dims, conf->fft_flags, fftmod_k, fftmod_k);

	complex float* fftmod_i = md_alloc(N, fft_idims, CFL_SIZE);
	md_resize_center(N, fft_idims, fftmod_i, fft_dims, fftmod_k, CFL_SIZE);

	long idims_normal[N];
	md_select_dims(N, ~(conf->batch_flags), idims_normal, data->idims);

	data->lop_fft_mod = linop_cdiag_create(N, idims_normal, conf->fft_flags, fftmod_i);
	md_free(fftmod_k);
	md_free(fftmod_i);

	return PTR_PASS(data);
}

static void mri_normal_operator_free_ops(struct mri_normal_operator_s* d)
{
	for (int i = 0; i < md_calc_size(d->N, d->bdims); i++) {

		operator_free(d->normal_ops[i]);
		d->normal_ops[i] = NULL;
	}
}

static void mri_normal_operator_data_free(struct mri_normal_operator_s* d)
{
	mri_normal_operator_free_ops(d);

	xfree(d->idims);
	xfree(d->pdims);
	xfree(d->cdims);
	xfree(d->kdims);

	xfree(d->bdims);

	linop_free(d->lop_fft);
	linop_free(d->lop_fft_mod);

	xfree(d->normal_ops);

	xfree(d);
}

static void mri_normal_operator_apply(struct mri_normal_operator_s* d, complex float* dst, const complex float* src)
{
	int N = d->N;
	unsigned long batch_flags = md_nontriv_dims(N, d->bdims);

	long istrs[N];
	md_calc_strides(N, istrs, d->idims, CFL_SIZE);

	long idims_normal[N];
	md_select_dims(d->N, ~(batch_flags), idims_normal, d->idims);


	long pos[d->N];
	for (int i = 0; i < d->N; i++)
		pos[i] = 0;

	do {
		const struct operator_s* normal_op = d->normal_ops[md_calc_offset(d->N, MD_STRIDES(d->N, d->bdims, 1), pos)];
		operator_apply(normal_op, d->N, idims_normal, &MD_ACCESS(d->N, istrs, pos, dst), d->N, idims_normal, &MD_ACCESS(d->N, istrs, pos, src));

	} while (md_next(d->N, d->bdims, ~(0ul), pos));
}








struct mri_normal_s {

	INTERFACE(nlop_data_t);

	struct mri_normal_operator_s* normal_ops;

	complex float* coil;
};

DEF_TYPEID(mri_normal_s);

static struct mri_normal_s* mri_normal_data_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	PTR_ALLOC(struct mri_normal_s, data);
	SET_TYPEID(mri_normal_s, data);

	data->normal_ops = mri_normal_operator_data_create(N, dims, idims, conf);
	data->coil = NULL;

	return PTR_PASS(data);
}

static void mri_normal_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_s, _data);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];

	bool der = nlop_der_requested(_data, 0, 0);

	if (der) {

		if (NULL == d->coil)
			d->coil = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->cdims, CFL_SIZE, coil);
		md_copy(d->normal_ops->N, d->normal_ops->cdims, d->coil, coil, CFL_SIZE);
	} else {

		md_free(d->coil);
		d->coil = NULL;
	}

	create_mri_normal_operator(d->normal_ops, der ? d->coil : coil, pattern);
	mri_normal_operator_apply(d->normal_ops, dst, image);
}

static void mri_normal_deradj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_s, _data);

	mri_normal_operator_apply(d->normal_ops, dst, src);
}

static void mri_normal_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_s, _data);

	mri_normal_operator_data_free(d->normal_ops);

	md_free(d->coil);
	xfree(d);
}

/**
 * Returns: MRI normal operator
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 *
 * Output tensors:
 * image:	idims: 	(Ix, Iy, Iz, Nc, Nb)
 */
const struct nlop_s* nlop_mri_normal_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	auto data = mri_normal_data_create(N, dims, idims, conf);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->normal_ops->idims);

	long nl_idims[3][N];
	md_copy_dims(N, nl_idims[0], data->normal_ops->idims);
	md_copy_dims(N, nl_idims[1], data->normal_ops->cdims);
	md_copy_dims(N, nl_idims[2], data->normal_ops->pdims);

	const struct nlop_s* result = nlop_generic_create(
			1, N, nl_odims, 3, N, nl_idims, CAST_UP(data),
			mri_normal_fun,
			(nlop_der_fun_t[3][1]){ { mri_normal_deradj }, { NULL }, { NULL } },
			(nlop_der_fun_t[3][1]){ { mri_normal_deradj }, { NULL }, { NULL } },
			NULL, NULL, mri_normal_del
		);

	return result;
}








/**
 * Returns operator computing gradient step
 * out = AH(A image - kspace)
 *
 * In non-cartesian case, the kspace is assumed to be gridded
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 *
 * Output tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 */

 const struct nlop_s* nlop_mri_gradient_step_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	auto nlop_normal = nlop_mri_normal_create(N, dims, idims, conf); // in: i, c, p
	auto nlop_adjoint = nlop_mri_adjoint_create(N, dims, idims, conf); // in: k, c, p

	auto result = nlop_chain2_FF(nlop_normal, 0, nlop_zaxpbz_create(N, idims, 1, -1), 0);
	result = nlop_chain2_FF(nlop_adjoint, 0, result, 0); //in: i, c, p, k, c, p

	result = nlop_dup_F(result, 1, 4);
	result = nlop_dup_F(result, 2, 4); //in: i, c, p, k

	result = nlop_permute_inputs_F(result, 4, (int[4]){0, 3, 1, 2});

	return result;
}







struct mri_normal_inversion_s {

	INTERFACE(nlop_data_t);

	int N_batch;
	struct mri_normal_operator_s* normal_ops;
	struct iter_conjgrad_conf* iter_conf;

	complex float* coil;
	complex float* out;

	bool store_tmp_adj;
	complex float* dout;	//Adjoint lambda and adjoint in
	complex float* AhAdout;	//share same intermediate result
};

static void mri_normal_inversion_set_normal_ops(struct mri_normal_inversion_s* d, const complex float* coil, const complex float* pattern, const complex float* lptr)
{
	complex float lambdas[d->N_batch];
	md_copy(d->normal_ops->N, d->normal_ops->bdims, lambdas, lptr, CFL_SIZE);

	for (int i = 0; i < d->N_batch; i++) {

		if ((0 != cimagf(lambdas[i])) || (0 > crealf(lambdas[i])))
			error("Lambda=%f+%fi is not non-negative real number!\n", crealf(lambdas[i]), cimagf(lambdas[i]));
		d->iter_conf[i].INTERFACE.alpha = crealf(lambdas[i]);
	}

	if (NULL == d->coil)
		d->coil = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->cdims, CFL_SIZE, coil);
	md_copy(d->normal_ops->N, d->normal_ops->cdims, d->coil, coil, CFL_SIZE);

	create_mri_normal_operator(d->normal_ops, d->coil, pattern);
}

DEF_TYPEID(mri_normal_inversion_s);

static void mri_normal_inversion(const struct mri_normal_inversion_s* d, complex float* dst, const complex float* src)
{
	int N = d->normal_ops->N;
	unsigned long batch_flags = md_nontriv_dims(N, d->normal_ops->bdims);

	long idims_normal[N];
	md_select_dims(N, ~batch_flags, idims_normal, d->normal_ops->idims);

	long istrs[N];
	md_calc_strides(N, istrs, d->normal_ops->idims, CFL_SIZE);

	long pos[N];
	for (int i = 0; i < N; i++)
		pos[i] = 0;

	do {
		int index = md_calc_offset(N, MD_STRIDES(N, d->normal_ops->bdims, 1), pos);

		auto normal_op = d->normal_ops->normal_ops[index];

		md_clear(N, idims_normal, &MD_ACCESS(N, istrs, pos, dst), CFL_SIZE);

		iter2_conjgrad(	CAST_UP(&(d->iter_conf[index])), normal_op,
				0, NULL, NULL, NULL, NULL,
				2 * md_calc_size(N, idims_normal),
				(float*)&MD_ACCESS(N, istrs, pos, dst),
				(const float*)&MD_ACCESS(N, istrs, pos, src),
				NULL);

	} while (md_next(N, d->normal_ops->bdims, ~(0ul), pos));
}

static void mri_inv_store_tmp_adj(struct mri_normal_inversion_s* d, const complex float* AhAdout, const complex float* dout)
{
	if (!d->store_tmp_adj)
		return;

	if (NULL == d->dout)
		d->dout = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->idims, CFL_SIZE, dout);
	if (NULL == d->AhAdout)
		d->AhAdout = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->idims, CFL_SIZE, AhAdout);

	md_copy(d->normal_ops->N, d->normal_ops->idims, d->dout, dout, CFL_SIZE);
	md_copy(d->normal_ops->N, d->normal_ops->idims, d->AhAdout, AhAdout, CFL_SIZE);
}

static bool mri_inv_load_tmp_adj(struct mri_normal_inversion_s* d, complex float* AhAdout, const complex float* dout)
{
	if ((NULL == d->dout) || (NULL == d->AhAdout))
		return false;

	if (0 != md_zrmse(d->normal_ops->N, d->normal_ops->idims, d->dout, dout))
		return false;

	md_copy(d->normal_ops->N, d->normal_ops->idims, AhAdout, d->AhAdout, CFL_SIZE);
	return true;
}

static void mri_normal_inversion_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, src);
}

static void mri_normal_inversion_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	if (mri_inv_load_tmp_adj(d, dst, src))
		return;

	mri_normal_inversion(d, dst, src);
	mri_inv_store_tmp_adj(d, dst, src);
}

static void mri_normal_inversion_der_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_inversion(d, dst, d->out);

	int N = d->normal_ops->N;
	const long* idims = d->normal_ops->idims;

	long istrs[N];
	md_calc_strides(N, istrs, d->normal_ops->idims, CFL_SIZE);
	long lstrs[N];
	md_calc_strides(N, lstrs, d->normal_ops->bdims, CFL_SIZE);

	md_zmul2(N, idims, istrs, dst, istrs, dst, lstrs, src);
	md_zsmul(N, idims, dst, dst, -1);
}

static void mri_normal_inversion_adj_lambda(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	int N = d->normal_ops->N;
	const long* idims = d->normal_ops->idims;
	const long* ldims = d->normal_ops->bdims;

	complex float* tmp = md_alloc_sameplace(N, idims, CFL_SIZE, dst);

	if (!mri_inv_load_tmp_adj(d, tmp, src)) {

		mri_normal_inversion(d, tmp, src);
		mri_inv_store_tmp_adj(d, tmp, src);
	}

	md_ztenmulc(N, ldims, dst, idims, d->out, idims, tmp);
	md_free(tmp);

	md_zsmul(N, ldims, dst, dst, -1);
	md_zreal(N, ldims, dst, dst);
}

static void mri_free_normal_ops(struct mri_normal_inversion_s* d)
{
	md_free(d->coil);
	d->coil = NULL;

	mri_normal_operator_free_ops(d->normal_ops);
}

static void mri_free_tmp_adj(struct mri_normal_inversion_s* d)
{
	md_free(d->dout);
	d->dout = NULL;
	md_free(d->AhAdout);
	d->AhAdout = NULL;
	d->store_tmp_adj = false;
}

static void mri_normal_inversion_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(5 == Narg);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];
	const complex float* lptr = args[4];

	bool der_in = nlop_der_requested(_data, 0, 0);
	bool der_lam = nlop_der_requested(_data, 3, 0);

	mri_free_tmp_adj(d);

	mri_normal_inversion_set_normal_ops(d, coil, pattern, lptr);

	mri_normal_inversion(d, dst, image);

	int N = d->normal_ops->N;
	const long* idims = d->normal_ops->idims;

	if (der_lam) {

		if(NULL == d->out)
			d->out = md_alloc_sameplace(N, idims, CFL_SIZE, dst);
		md_copy(N, idims, d->out, dst, CFL_SIZE);
	} else {

		md_free(d->out);
		d->out = NULL;

		if (!der_in)
			mri_free_normal_ops(d);
	}

	d->store_tmp_adj = der_lam && der_in;
}

static void mri_normal_inversion_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	mri_normal_operator_data_free(d->normal_ops);

	md_free(d->coil);

	xfree(d->iter_conf);

	md_free(d->out);

	md_free(d->dout);
	md_free(d->AhAdout);

	xfree(d);
}

static struct mri_normal_inversion_s* mri_normal_inversion_data_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	PTR_ALLOC(struct mri_normal_inversion_s, data);
	SET_TYPEID(mri_normal_inversion_s, data);

	data->normal_ops = mri_normal_operator_data_create(N, dims, idims, conf);
	data->coil = NULL;

	// will be initialized later, to transparently support GPU
	data->out= NULL;
	data->coil = NULL;

	data->dout = NULL;
	data->AhAdout = NULL;
	data->store_tmp_adj = false;

	data->N_batch = md_calc_size(data->normal_ops->N, data->normal_ops->bdims);
	data->iter_conf = *TYPE_ALLOC(struct iter_conjgrad_conf[data->N_batch]);

	for (int i = 0; i < data->N_batch; i++) {

		if (NULL == iter_conf) {

			data->iter_conf[i] = iter_conjgrad_defaults;
			data->iter_conf[i].l2lambda = 1.;
			data->iter_conf[i].maxiter = 50;
		} else {

			data->iter_conf[i] = *iter_conf;
		}
	}

	return PTR_PASS(data);
}

/**
 * Create an operator applying the inverse normal mri forward model on its input
 *
 * out = (A^HA +l1)^-1 in
 * A = Pattern FFT Coils
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 * @param lammbda_fixed if -1, lambda is an input of the nlop
 *
 * Input tensors:
 * image:	idims: 	(Ix, Iy, Iz,  1, Nb)
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz,  1,  1 / Nb)
 * lambda:	ldims:	( 1,  1,  1,  1, Nb)
 *
 * Output tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 */
const struct nlop_s* mri_normal_inversion_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	auto data = mri_normal_inversion_data_create(N, dims, idims, conf, iter_conf);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->normal_ops->idims);

	long nl_idims[4][N];
	md_copy_dims(N, nl_idims[0], data->normal_ops->idims);
	md_copy_dims(N, nl_idims[1], data->normal_ops->cdims);
	md_copy_dims(N, nl_idims[2], data->normal_ops->pdims);
	md_copy_dims(N, nl_idims[3], data->normal_ops->bdims);

	const struct nlop_s* result = nlop_generic_create(	1, N, nl_odims, 4, N, nl_idims, CAST_UP(data),
									mri_normal_inversion_fun,
									(nlop_der_fun_t[4][1]){ { mri_normal_inversion_der }, { NULL }, { NULL }, { mri_normal_inversion_der_lambda } },
									(nlop_der_fun_t[4][1]){ { mri_normal_inversion_adj }, { NULL }, { NULL }, { mri_normal_inversion_adj_lambda } },
									NULL, NULL, mri_normal_inversion_del
								);

	return result;
}








static void mri_reg_proj_der(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	complex float* tmp = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->idims, CFL_SIZE, dst);

	mri_normal_operator_apply(d->normal_ops, tmp, src);
	mri_normal_inversion(d, dst, tmp);

	md_free(tmp);

}

static void mri_reg_proj_adj(const nlop_data_t* _data, unsigned int o, unsigned int i, complex float* dst, const complex float* src)
{
	UNUSED(o);
	UNUSED(i);

	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);

	complex float* tmp = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->idims, CFL_SIZE, dst);

	if (!mri_inv_load_tmp_adj(d, tmp, src)) {

		mri_normal_inversion(d, tmp, src);
		mri_inv_store_tmp_adj(d, tmp, src);
	}

	mri_normal_operator_apply(d->normal_ops, dst, tmp);

	md_free(tmp);

}

static void mri_reg_proj_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_inversion_s, _data);
	assert(4 == Narg);

	complex float* dst = args[0];
	const complex float* image = args[1];
	const complex float* coil = args[2];
	const complex float* pattern = args[3];
	const complex float* lptr = args[4];

	bool der_in = nlop_der_requested(_data, 0, 0);
	bool der_lam = nlop_der_requested(_data, 3, 0);

	mri_free_tmp_adj(d);

	mri_normal_inversion_set_normal_ops(d, coil, pattern, lptr);

	complex float* tmp = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->idims, CFL_SIZE, dst);
	mri_normal_operator_apply(d->normal_ops, tmp, image);
	mri_normal_inversion(d, dst, tmp);

	md_free(tmp);

	if (der_lam) {

		if(NULL == d->out)
			d->out = md_alloc_sameplace(d->normal_ops->N, d->normal_ops->idims, CFL_SIZE, dst);
		md_copy(d->normal_ops->N, d->normal_ops->idims, d->out, dst, CFL_SIZE);
	} else {

		md_free(d->out);
		d->out = NULL;

		if (!der_in)
			mri_free_normal_ops(d);
	}
	d->store_tmp_adj = der_lam && der_in;
}

 /**
 * Create an operator projecting its input to the kernel of the mri forward operator (regularized with lambda)
 *
 * out = (id - (A^HA +l1)^-1A^HA) in
 * A = Pattern FFT Coils
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 *
 * Input tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 / Nb)
 * lambda:	ldims:	( 1,  1,  1,  1, Nb)
 *
 * Output tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 */
const struct nlop_s* mri_reg_proj_ker_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	auto data = mri_normal_inversion_data_create(N, dims, idims, conf, iter_conf);

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->normal_ops->idims);

	long nl_idims[4][N];
	md_copy_dims(N, nl_idims[0], data->normal_ops->idims);
	md_copy_dims(N, nl_idims[1], data->normal_ops->cdims);
	md_copy_dims(N, nl_idims[2], data->normal_ops->pdims);
	md_copy_dims(N, nl_idims[3], data->normal_ops->bdims);

	const struct nlop_s* result = nlop_generic_create(	1, N, nl_odims, 4, N, nl_idims, CAST_UP(data),
									mri_reg_proj_fun,
									(nlop_der_fun_t[4][1]){ { mri_reg_proj_der }, { NULL }, { NULL }, { mri_normal_inversion_der_lambda } },
									(nlop_der_fun_t[4][1]){ { mri_reg_proj_adj }, { NULL }, { NULL }, { mri_normal_inversion_adj_lambda } },
									NULL, NULL, mri_normal_inversion_del
								);

	result = nlop_chain2_FF(result, 0, nlop_zaxpbz_create(N, nl_idims[0], -1., 1.), 0);
	result = nlop_dup_F(result, 0, 1);

	return result;

}








/**
 * Create an operator computing the Tickhonov regularized pseudo-inverse of the MRI operator
 *
 * out = [(1 + lambda)](A^HA +l1)^-1 A^Hin
 * A = Pattern FFT Coils
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 * @param iter_conf configuration for conjugate gradient
 * @param lammbda_fixed if -1, lambda is an input of the nlop
 *
 * Input tensors:
 * kspace:	kdims: 	(Nx, Ny, Nz, Nc, Nb)
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 * lambda:	ldims:	( 1,  1,  1,  1, Nb)
 *
 * Output tensors:
 * image:	idims: 	(Ix, Iy, Iz, 1,  Nb)
 */

const struct nlop_s* mri_reg_pinv(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf, struct iter_conjgrad_conf* iter_conf)
{
	auto nlop_zf = nlop_mri_adjoint_create(N, dims, idims, conf);// in: kspace, coil, pattern; out: Atb
	auto nlop_norm_inv = mri_normal_inversion_create(N, dims, idims, conf, iter_conf); // in: Atb, coil, pattern, [lambda]; out: A^+b

	auto nlop_pseudo_inv = nlop_chain2_swap_FF(nlop_zf, 0, nlop_norm_inv, 0); // in: kspace, coil, pattern, coil, pattern, [lambda]; out: A^+b
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 1, 3);
	nlop_pseudo_inv = nlop_dup_F(nlop_pseudo_inv, 2, 3);// in: kspace, coil, pattern, [lambda]; out: A^+b

	return nlop_pseudo_inv;
}









struct mri_normal_power_iter_s {

	INTERFACE(nlop_data_t);

	struct mri_normal_operator_s* normal_ops;

	complex float* noise;
	int max_iter;
};

DEF_TYPEID(mri_normal_power_iter_s);

static void mri_normal_power_iter_del(const nlop_data_t* _data)
{
	const auto d = CAST_DOWN(mri_normal_power_iter_s, _data);

	mri_normal_operator_data_free(d->normal_ops);

	md_free(d->noise);

	xfree(d);
}

static void mri_normal_power_iter_fun(const nlop_data_t* _data, int Narg, complex float* args[Narg])
{
	const auto d = CAST_DOWN(mri_normal_power_iter_s, _data);

	assert(3 == Narg);

	complex float* dst = args[0];
	const complex float* coil = args[1];
	const complex float* pattern = args[2];

	int N = d->normal_ops->N;
	unsigned long batch_flags = md_nontriv_dims(N, d->normal_ops->bdims);

	long idims[N];
	md_select_dims(N, ~batch_flags, idims, d->normal_ops->idims);

	if (NULL == d->noise) {

		d->noise = md_alloc_sameplace(N, idims, CFL_SIZE, dst);
		md_gaussian_rand(N, idims, d->noise);
	}

	int N_batch = md_calc_size(N, d->normal_ops->bdims);
	complex float max_eigen[N_batch];

	create_mri_normal_operator(d->normal_ops, coil, pattern);

	for (int i = 0; i < N_batch; i++) {

		auto normal_op = d->normal_ops->normal_ops[i];

		long size = md_calc_size(N, idims);
		max_eigen[i] = iter_power(d->max_iter, normal_op, 2 * size, (float*)d->noise);
	}

	md_copy(1, MD_DIMS(N_batch), dst, max_eigen, CFL_SIZE);

	mri_normal_operator_free_ops(d->normal_ops);
}


/**
 * Returns: Operator estimating the max eigen value of mri normal operator
 *
 * @param N
 * @param dims 	kspace dimension (possibly oversampled)
 * @param idims image dimensions
 * @param conf can be NULL to fallback on nlop_mri_simple
 *
 *
 * for default dims:
 *
 * Input tensors:
 * coil:	cdims:	(Ix, Iy, Iz, Nc, Nb)
 * pattern:	pdims:	(Nx, Ny, Nz, 1,  1 )
 *
 * Output tensors:
 * max eigen:	ldims: 	( 1,  1,  1,  1, Nb)
 */
const struct nlop_s* mri_normal_max_eigen_create(int N, const long dims[N], const long idims[N], const struct config_nlop_mri_s* conf)
{
	assert(test_idims_compatible(N, dims, idims, conf));

	if (NULL == conf)
		conf = &conf_nlop_mri_simple;

	PTR_ALLOC(struct mri_normal_power_iter_s, data);
	SET_TYPEID(mri_normal_power_iter_s, data);

	data->normal_ops = mri_normal_operator_data_create(N, dims, idims, conf);

	data->noise = NULL;
	data->max_iter = 30;

	long nl_odims[1][N];
	md_copy_dims(N, nl_odims[0], data->normal_ops->bdims);

	long nl_idims[2][N];
	md_copy_dims(N, nl_idims[0], data->normal_ops->cdims);
	md_copy_dims(N, nl_idims[1], data->normal_ops->pdims);

	const struct nlop_s* result = nlop_generic_create(
			1, N, nl_odims, 2, N, nl_idims, CAST_UP(PTR_PASS(data)),
			mri_normal_power_iter_fun,
			NULL, NULL,
			NULL, NULL, mri_normal_power_iter_del
		);

	return result;
}

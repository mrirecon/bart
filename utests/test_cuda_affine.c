#include "misc/debug.h"
#include "misc/misc.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include "num/rand.h"

#include "linops/lintest.h"

#include "nlops/nlop.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/nltest.h"

#include "motion/affine.h"
#include "motion/interpolate.h"

#include "utest.h"


static bool test_affine_rot_transpose(void)
{
	auto nlop_rot = nlop_affine_rotation_2D();
	complex float pars[1] = { 0.5 * M_PI };
	complex float trafo[12];
	nlop_apply(nlop_rot, 2, MD_DIMS(3, 4), trafo, 1, MD_DIMS(1), pars);

	long dims[4] =  { 3, 3, 1, 1 };
	long cdims[4] = { 3, 3, 1, 3 };

	const struct nlop_s* nlop_interp = nlop_interpolate_create(3, 7, 1, false, 4, dims, cdims, dims);
	nlop_interp = nlop_prepend_FF(nlop_affine_compute_pos(3, 4, dims, dims, nlop_rot), nlop_interp, 1);
	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);


	complex float* inp = md_alloc(4, dims, CFL_SIZE);
	complex float* out = md_alloc(4, dims, CFL_SIZE);
	complex float* tmp = md_alloc(4, dims, CFL_SIZE);

	md_gaussian_rand(4, dims, inp);

	nlop_generic_apply_unchecked(nlop_interp, 3, (void*[3]) { out, inp, pars });

	md_transpose(4, 0, 1, dims, tmp, dims, inp, CFL_SIZE);
	md_flip(4, dims, 2, inp, tmp, CFL_SIZE);

	float err = md_znrmse(4, dims, out, inp);

	md_free(inp);
	md_free(out);
	md_free(tmp);

	nlop_free(nlop_interp);

	UT_RETURN_ASSERT(UT_TOL > err);
}


UT_GPU_REGISTER_TEST(test_affine_rot_transpose);



static bool test_affine_nlop_rot2D(void)
{
	auto nlop_rot = nlop_affine_rotation_2D();
	nlop_rot = nlop_gpu_wrapper_F(nlop_rot);


	complex float pars[1] = { 0.1 * M_PI };

	float err = nlop_test_derivative_at(nlop_rot, pars);
	
	UT_RETURN_ON_FAILURE(UT_TOL > nlop_test_adj_derivatives(nlop_rot, true));

	nlop_free(nlop_rot);

	UT_RETURN_ASSERT(0.1 > err);
}

UT_GPU_REGISTER_TEST(test_affine_nlop_rot2D);



static bool test_affine_nlop_interpolate(void)
{
	long dims[4] = { 8, 8, 1, 1 };
	long cdims[4] = { 8, 8, 1, 3 };

	auto nlop_rot = nlop_affine_rotation_2D();
	const struct nlop_s* nlop_interp = nlop_interpolate_create(3, 7, 1, false, 4, dims, cdims, dims);
	nlop_interp = nlop_prepend_FF(nlop_affine_compute_pos(3, 4, dims, dims, nlop_rot), nlop_interp, 1);

	complex float pars[1] = { 0.2 * M_PI };
	nlop_interp = nlop_set_input_const_F(nlop_interp, 1, 1, MD_DIMS(1), true, pars);
	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);

	float err = nlop_test_derivatives(nlop_interp);

	UT_RETURN_ON_FAILURE(1.E-5 > linop_test_adjoint_real(nlop_get_derivative(nlop_interp, 0, 0)));

	nlop_free(nlop_interp);

	UT_RETURN_ASSERT(0.001 > err);
}

UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate);

static bool test_affine_nlop_interpolate_coord(void)
{
	num_rand_init(123);

	long dims[4] = { 8, 8, 1, 1 };
	long cdims[4] = { 8, 8, 1, 3 };

	auto nlop_rot = nlop_affine_rotation_2D();

	const struct nlop_s* nlop_interp = nlop_interpolate_create(3, 7, 1, false, 4, dims, cdims, dims);
	nlop_interp = nlop_prepend_FF(nlop_affine_compute_pos(3, 4, dims, dims, nlop_rot), nlop_interp, 1);

	complex float* tmp = md_alloc(4, dims, CFL_SIZE);
	md_gaussian_rand(4, dims, tmp);
	nlop_interp = nlop_set_input_const_F(nlop_interp, 0, 4, dims, true, tmp);
	md_free(tmp);

	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);

	complex float pars[1] = { 0.1 * M_PI };
	
	
	float err = nlop_test_derivative_at(nlop_interp, pars);

	UT_RETURN_ON_FAILURE(1.E-5 > linop_test_adjoint_real(nlop_get_derivative(nlop_interp, 0, 0)));

	nlop_free(nlop_interp);

	UT_RETURN_ASSERT(0.01 > err);
}

UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate_coord);


static bool test_affine_nlop_interpolate_cood_points(void)
{
	long idims[4] = { 5, 5, 5, 1 };
	long odims[4] = { 2, 1, 1, 1 };
	long cdims[4] = { 2, 1, 1, 3 };

	//derivative is not defined on grid points
	complex float coor[6] = { 0.2, 0.9, 0.1, 1.2, 4.9, 0.1 };
	
	const struct nlop_s* nlop_interp = nlop_interpolate_create(3, 7, 1, false, 4, odims, cdims, idims);
	
	complex float* tmp = md_alloc(4, idims, CFL_SIZE);
	md_gaussian_rand(4, idims, tmp);
	nlop_interp = nlop_set_input_const_F(nlop_interp, 0, 4, idims, true, tmp);
	md_free(tmp);

	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);

	float err = nlop_test_derivative_at(nlop_interp, coor);

	UT_RETURN_ON_FAILURE(1.E-5 > linop_test_adjoint_real(nlop_get_derivative(nlop_interp, 0, 0)));

	nlop_free(nlop_interp);

	UT_RETURN_ASSERT(0.1 > err);
}

UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate_cood_points);



static bool test_affine_nlop_interpolate_cood_points_keys(void)
{
	long idims[4] = { 5, 5, 5, 1 };
	long odims[4] = { 2, 1, 1, 1 };
	long cdims[4] = { 2, 1, 1, 3 };

	//derivative is not defined on grid points
	complex float coor[6] = { 0.2, 0.9, 0.0, 1.2, 4.9, 0.1 };
	
	const struct nlop_s* nlop_interp = nlop_interpolate_create(3, 7, 3, false, 4, odims, cdims, idims);
	
	complex float* tmp = md_alloc(4, idims, CFL_SIZE);
	md_gaussian_rand(4, idims, tmp);
	nlop_interp = nlop_set_input_const_F(nlop_interp, 0, 4, idims, true, tmp);
	md_free(tmp);

	nlop_interp = nlop_gpu_wrapper_F(nlop_interp);

	float err = nlop_test_derivative_at(nlop_interp, coor);

	UT_RETURN_ON_FAILURE(1.E-5 > linop_test_adjoint_real(nlop_get_derivative(nlop_interp, 0, 0)));

	nlop_free(nlop_interp);

	debug_printf(DP_WARN, "%e\n", err);

	UT_RETURN_ASSERT(0.1 > err);
}

UT_GPU_REGISTER_TEST(test_affine_nlop_interpolate_cood_points_keys);



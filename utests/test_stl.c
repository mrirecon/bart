/* Copyright 2024. University Medical Center Göttingen, Germany
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Martin Heide
 */

#include <stdbool.h>
#include <complex.h>
#include <math.h>

#include "utest.h"

#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/linalg.h"

#include "stl/misc.h"
#include "stl/models.h"

static bool test_stl_normal_vector(void)
{
        long dims[DIMS], strs[DIMS], pos[DIMS];
        md_set_dims(DIMS, pos, 0);

        double* model = stl_internal_tetrahedron(DIMS, dims);

        md_calc_strides(DIMS, strs, dims, DL_SIZE);

        stl_compute_normals(DIMS, dims, model);

        double d0 = 0.577350269189626;
        double d1 = -0.577350269189626;
        double n0[3] = { d0, d0, d1 };
        double n1[3] = { d0, d1, d0 };
        double n2[3] = { d1, d0, d0 };
        double n3[3] = { d1, d1, d1 };

        double o[3];
        bool b = true;
        double* d;

        pos[1] = 3;
        pos[2] = 0;
        d = &MD_ACCESS(DIMS, strs, pos, model);
        vec3d_saxpy(o, n0, -1, d);
        b = b && (TOL > vec3d_norm(o));

        pos[2] = 1;
        d = &MD_ACCESS(DIMS, strs, pos, model);
        vec3d_saxpy(o, n1, -1, d);
        b = b && (TOL > vec3d_norm(o));

        pos[2] = 2;
        d = &MD_ACCESS(DIMS, strs, pos, model);
        vec3d_saxpy(o, n2, -1, d);
        b = b && (TOL > vec3d_norm(o));

        pos[2] = 3;
        d = &MD_ACCESS(DIMS, strs, pos, model);
	vec3d_saxpy(o, n3, -1, d);
        b = b && (TOL > vec3d_norm(o));

        md_free(model);
        return b;
}

UT_REGISTER_TEST(test_stl_normal_vector);

static bool test_stl_cfl_double_conversion(void)
{
        bool b = true;
        long dims[DIMS];
        double* model = stl_internal_tetrahedron(DIMS, dims);

        complex float* cmodel = md_alloc(DIMS, dims, CFL_SIZE);
        stl_d2cfl(DIMS, dims, model, cmodel);

        double* model0 = stl_cfl2d(DIMS, dims, cmodel);

        complex float* cmodel0 = md_alloc(DIMS, dims, CFL_SIZE);
        stl_d2cfl(DIMS, dims, model0, cmodel0);

        complex float* s = md_alloc(DIMS, dims, CFL_SIZE);
        md_zsub(DIMS, dims, s, cmodel, cmodel0);

        complex float r;
        md_zsum(DIMS, dims, ~0UL, &r, s);
        if (0 < sqrt(creal(r) * creal(r) + cimag(r) * cimag(r)))
                b = false;
        long pos[DIMS], strs[DIMS];
        md_set_dims(DIMS, pos, 0);
        md_calc_strides(DIMS, strs, dims, DL_SIZE);

        double d = 0;
        do {
                d += MD_ACCESS(DIMS, strs, pos, model) - MD_ACCESS(DIMS, strs, pos, model0);
        } while(md_next(DIMS, dims, ~0UL, pos));

        if (0 < d)
                b = false;

        md_free(s);
        md_free(cmodel0);
        md_free(model0);
        md_free(cmodel);
        md_free(model);

        return b;
}

UT_REGISTER_TEST(test_stl_cfl_double_conversion);

static bool vec_finite(int N, const double* v)
{
	for (int i = 0; i < N; i++)
		if (INFINITY == v[i])
			return false;

	return true;
}

static bool check_triangle(const struct triangle* t)
{
	if (!(vec_finite(3, t->v0) &&
		vec_finite(3, t->v1) &&
		vec_finite(3, t->v2) &&
		vec_finite(3, t->n) &&
		vec_finite(3, t->e0) &&
		vec_finite(3, t->e1) &&
		vec_finite(3, t->ctr) &&
		vec_finite(3, t->rot) &&
		vec_finite(6, t->poly) &&
		INFINITY != t->svol &&
		INFINITY != t->angle))
		return false;

	return true;
}

static bool test_stlgeomprocessing(void)
{
        bool b = true;
        long dims[DIMS], strs[DIMS], pos[DIMS];

        double* model = stl_internal_tetrahedron(DIMS, dims);

	struct triangle_stack ts = triangle_stack_defaults;

	ts.N = dims[2];

	if (ts.N != 4)
		b = false;

	ts.tri = xmalloc((size_t) ts.N * sizeof(struct triangle));

	struct triangle* t = ts.tri;

	md_calc_strides(DIMS, strs, dims, DL_SIZE);
	md_set_dims(DIMS, pos, 0);

	for (pos[2] = 0; pos[2] < ts.N; pos[2]++) {

		t[pos[2]] = triangle_defaults;

		memcpy(&t[pos[2]], &MD_ACCESS(DIMS, strs, pos, model), 12 * DL_SIZE);

		stl_relative_position(&t[pos[2]]);

		if (!check_triangle(&t[pos[2]]))
			b = false;

		if (1E-14 < fabs(0.06075 - t[pos[2]].svol))
			b = false;
	}

	if (1E-14 < fabs(-0.19019237886466838772 - t[0].poly[0]) ||
		1E-14 < fabs(0.70980762113533146795 - t[0].poly[1]) ||
		1E-14 < fabs(-0.51961524227066313575 - t[0].poly[2]) ||
		1E-14 < fabs(-0.51961524227066313575 - t[0].poly[3]) ||
		1E-14 < fabs(0.70980762113533146795 - t[0].poly[4]) ||
		1E-14 < fabs(-0.19019237886466838772 - t[0].poly[5]))
		b = false;


	md_free(ts.tri);
        md_free(model);

        return b;
}

UT_REGISTER_TEST(test_stlgeomprocessing);

static bool test_stl_triangles_on_axes(void)
{
        bool b = true;

	struct triangle t = triangle_defaults;

	t.v0[0] = 1;
	t.v0[1] = 0;
	t.v0[2] = 0;
	t.v1[0] = 1;
	t.v1[1] = 1;
	t.v1[2] = 0;
	t.v2[0] = 0;
	t.v2[1] = 1;
	t.v2[2] = 0;
	t.n[0] = 0;
	t.n[1] = 0;
	t.n[2] = 1;

	stl_relative_position(&t);

	if (1E-14 < fabs(t.angle))
		b = false;

	t.n[2] = -1;

	stl_relative_position(&t);

	if (1E-14 < fabs(M_PI - t.angle))
		b = false;

	t.v0[0] = 0;
	t.v0[1] = 1;
	t.v0[2] = 0;
	t.v1[0] = 0;
	t.v1[1] = 1;
	t.v1[2] = 1;
	t.v2[0] = 0;
	t.v2[1] = 0;
	t.v2[2] = 1;
	t.n[0] = 1;
	t.n[1] = 0;
	t.n[2] = 0;

	stl_relative_position(&t);

	if (1E-14 < fabs(M_PI - 2 * t.angle))
		b = false;

	t.n[0] = -1;

	stl_relative_position(&t);

	if (1E-14 < fabs(M_PI - 2 * t.angle))
		b = false;

	t.v0[0] = 1;
	t.v0[1] = 0;
	t.v0[2] = 0;
	t.v1[0] = 1;
	t.v1[1] = 0;
	t.v1[2] = 1;
	t.v2[0] = 0;
	t.v2[1] = 0;
	t.v2[2] = 1;
	t.n[0] = 0;
	t.n[1] = 1;
	t.n[2] = 0;

	stl_relative_position(&t);

	if (1E-14 < fabs(M_PI - 2 * t.angle))
		b = false;

	t.n[1] = -1;

	stl_relative_position(&t);

	if (1E-14 < fabs(M_PI - 2 * t.angle))
		b = false;

        return b;
}

UT_REGISTER_TEST(test_stl_triangles_on_axes);

static bool test_stl_measures(void)
{
        bool b = true;

	long dimshex[DIMS];
	long dimstet[DIMS];

	double* mhex = stl_internal_hexahedron(DIMS, dimshex);
	double* mtet = stl_internal_tetrahedron(DIMS, dimstet);

	struct triangle_stack* tshex = stl_preprocess_model(DIMS, dimshex, mhex);
	struct triangle_stack* tstet = stl_preprocess_model(DIMS, dimstet, mtet);

	struct triangle* thex = tshex->tri;
	struct triangle* ttet = tstet->tri;

	double smvh = 0;
	double vmvh = 0;
	double smvt = 0;
	double vmvt = 0;

	for (int i = 0; i < tshex->N; i++) {

		smvh += thex[i].sur;
		vmvh += thex[i].svol;
	}
	for (int i = 0; i < tstet->N; i++) {

		smvt += ttet[i].sur;
		vmvt += ttet[i].svol;
	}

	if (TOL < fabs(4.86 - smvh))
		b = false;

	if (TOL < fabs(0.729 - vmvh))
		b = false;

	if (TOL < fabs(2.80592230826158139934 - smvt))
		b = false;

	if (TOL < fabs(0.243 - vmvt))
		b = false;

	md_free(tshex->tri);
	md_free(tshex);
	md_free(tstet->tri);
	md_free(tstet);
	md_free(mhex);
	md_free(mtet);

        return b;
}

UT_REGISTER_TEST(test_stl_measures);

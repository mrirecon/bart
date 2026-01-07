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
        double n0[3] = {d0, d0, d1};
        double n1[3] = {d0, d1, d0};
        double n2[3] = {d1, d0, d0};
        double n3[3] = {d1, d1, d1};

        double o[3];
        bool b = true;
        double* d;

        pos[1] = 3;
        pos[2] = 0;
        d = &MD_ACCESS(DIMS, strs, pos, model);
        stl_sub_vec3(o, n0, d);
        b = b && (TOL > stl_norm_vec(3, o));

        pos[2] = 1;
        d = &MD_ACCESS(DIMS, strs, pos, model);
        stl_sub_vec3(o, n1, d);
        b = b && (TOL > stl_norm_vec(3, o));

        pos[2] = 2;
        d = &MD_ACCESS(DIMS, strs, pos, model);
        stl_sub_vec3(o, n2, d);
        b = b && (TOL > stl_norm_vec(3, o));

        pos[2] = 3;
        d = &MD_ACCESS(DIMS, strs, pos, model);
        stl_sub_vec3(o, n3, d);
        b = b && (TOL > stl_norm_vec(3, o));

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

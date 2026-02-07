/* Copyright 2025. University Medical Center Göttingen, Germany
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Martin Heide
 */

#include "models.h"
#include "misc/debug.h"
#include "misc/misc.h"
#include "num/multind.h"
#include "num/flpmath.h"

#include "stl/misc.h"

// direction of normal is outward.
// we don't store the normal vector bc it will be computed in the stl_internal_* function.
// vertices listed counterclockwise when looked on from outside (right-hand rule).
//
// convert vertices in array to contiguous fortran data format and compute normal vector
static double* stl_internal_model(long dims[3], const void* v)
{
        if (3 != dims[0] || 4 != dims[1])
                error("dimensions do not match dimensions for stl format");

        const double* arr = v;

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

        double* model = md_alloc(3, dims, DL_SIZE);

        long pos[3] = { };

        do {

                // Normal vector is computed afterwards.
                if (3 > pos[1])
                        MD_ACCESS(3, strs, pos, model) = arr[pos[2] * dims[0] * (dims[1] - 1) + pos[1] * dims[0] + pos[0]];

        } while (md_next(3, dims, ~0UL, pos));

        stl_compute_normals(dims, model);

        return model;
}

static const double stl_hexahedron[12][3][3] = {

        { { 0.45, 0.45, -0.45 }, { -0.45, 0.45, -0.45 }, { 0.45, 0.45, 0.45 } },
        { { 0.45, 0.45, 0.45 }, { -0.45, 0.45, -0.45 }, { -0.45, 0.45, 0.45 } },
        { { 0.45, -0.45, -0.45 }, { 0.45, -0.45, 0.45 }, { -0.45, -0.45, 0.45 } },
        { { 0.45, -0.45, -0.45 }, { -0.45, -0.45, 0.45 }, { -0.45, -0.45, -0.45 } },
        { { 0.45, 0.45, -0.45 }, { 0.45, 0.45, 0.45 }, { 0.45, -0.45, 0.45 } },
        { { 0.45, 0.45, -0.45 }, { 0.45, -0.45, 0.45 }, { 0.45, -0.45, -0.45 } },
        { { -0.45, 0.45, -0.45 }, { -0.45, -0.45, -0.45 }, { -0.45, 0.45, 0.45 } },
        { { -0.45, 0.45, 0.45 }, { -0.45, -0.45, -0.45 }, { -0.45, -0.45, 0.45 } },
        { { -0.45, 0.45, 0.45 }, { 0.45, -0.45, 0.45 }, { 0.45, 0.45, 0.45 } },
        { { -0.45, 0.45, 0.45 }, { -0.45, -0.45, 0.45 }, { 0.45, -0.45, 0.45 } },
        { { -0.45, -0.45, -0.45 }, { -0.45, 0.45, -0.45 }, { 0.45, 0.45, -0.45 } },
        { { -0.45, -0.45, -0.45 }, { 0.45, 0.45, -0.45 }, { 0.45, -0.45, -0.45 } },
};

double* stl_internal_hexahedron(long dims[3])
{
        dims[0] = 3;
        dims[1] = 4;
        dims[2] = 12;

        return stl_internal_model(dims, stl_hexahedron);
}


static const double stl_tetrahedron[4][3][3] = {

        { { -0.45, 0.45, -0.45 }, { 0.45, 0.45, 0.45 }, { 0.45, -0.45, -0.45 } },
        { { -0.45, -0.45, 0.45 }, { 0.45, -0.45, -0.45 }, { 0.45, 0.45, 0.45 } },
        { { -0.45, -0.45, 0.45 }, { 0.45, 0.45, 0.45 }, { -0.45, 0.45, -0.45 } },
        { { -0.45, -0.45, 0.45 }, { -0.45, 0.45, -0.45 }, { 0.45, -0.45, -0.45 } },
};

double* stl_internal_tetrahedron(long dims[3])
{
        dims[0] = 3;
        dims[1] = 4;
        dims[2] = 4;

        return stl_internal_model(dims, stl_tetrahedron);
}




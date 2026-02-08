/* Copyright 2025. University Medical Center Göttingen.
 * Copyright 2026. Institute of Biomedical Imaging. TU Graz.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2024 Martin Heide
 */

#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>

#include "num/multind.h"
#include "num/linalg.h"
#include "num/flpmath.h"

#include "misc/io.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/version.h"

#include "stl/misc.h"

struct triangle triangle_defaults = {

	.v0 = { INFINITY, INFINITY, INFINITY },
	.v1 = { INFINITY, INFINITY, INFINITY },
	.v2 = { INFINITY, INFINITY, INFINITY },
	.n = { INFINITY, INFINITY, INFINITY },
	.e0 = { INFINITY, INFINITY, INFINITY },
	.e1 = { INFINITY, INFINITY, INFINITY },
	.ctr = { INFINITY, INFINITY, INFINITY },
	.rot = { INFINITY, INFINITY, INFINITY },
	.angle = INFINITY,
	.svol = INFINITY,
	.poly = { INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY },
	.sur = INFINITY,
};

struct triangle_stack triangle_stack_defaults = {

	.N = -1,
	.tri = NULL,
};


// compute minimal and maximal vertex coordinates
static void stl_coordinate_limits(const long dims[3], const double* model, double* min_v, double* max_v)
{
        min_v[0] = INFINITY;
        min_v[1] = INFINITY;
        min_v[2] = INFINITY;
        max_v[0] = -INFINITY;
        max_v[1] = -INFINITY;
        max_v[2] = -INFINITY;

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);
        long pos[3];
        md_set_dims(3, pos, 0);

        long pos0[3], pos1[3], pos2[3];
        md_set_dims(3, pos0, 0);
        md_set_dims(3, pos1, 0);
        md_set_dims(3, pos2, 0);
        pos1[1] = 1;
        pos2[1] = 2;

        for (int i = 0; i < dims[2]; i++) {

                pos0[2] = i;
                pos1[2] = i;
                pos2[2] = i;

                for (int j = 0; j < dims[0]; j++) {

                        pos0[0] = j;
                        pos1[0] = j;
                        pos2[0] = j;

                        double v0 = MD_ACCESS(3, strs, pos0, model);
                        double v1 = MD_ACCESS(3, strs, pos1, model);
                        double v2 = MD_ACCESS(3, strs, pos2, model);

                        if (min_v[j] > v0)
                                min_v[j] = v0;

                        if (min_v[j] > v1)
                                min_v[j] = v1;

                        if (min_v[j] > v2)
                                min_v[j] = v2;

                        if (max_v[j] < v0)
                                max_v[j] = v0;

                        if (max_v[j] < v1)
                                max_v[j] = v1;

                        if (max_v[j] < v2)
                                max_v[j] = v2;
                }
        }
}

// Scales all vertex coordinates by scale vector. It doesnt scale the normal vector.
void stl_scale_model(const long dims[3], double* model, const double scale[3])
{
        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos[3];
                md_set_dims(3, pos, 0);
                pos[2] = i;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        for (pos[1] = 0; pos[1] < dims[1] - 1; pos[1]++)
                                MD_ACCESS(3, strs, pos, model) *= scale[pos[0]];
        }
}

// Shifts all vertex coordinates by shift vector. It doesn't shift the normal vector (shift invariant)
void stl_shift_model(const long dims[3], double* model, const double shift[3])
{
        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos[3];
                md_set_dims(3, pos, 0);
                pos[2] = i;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        for (pos[1] = 0; pos[1] < dims[1] - 1; pos[1]++)
                                MD_ACCESS(3, strs, pos, model) += shift[pos[0]];
        }
}

// shift and scale the model to FOV of size fov_size > 0.
void stl_center_fov(const long dims[3], double* model, double fov_size)
{
        if (0 >= fov_size)
                error("fov_size should be positive.");

        double min_v[3];
	double max_v[3];
        stl_coordinate_limits(dims, model, min_v, max_v);

        double crange[3] = { max_v[0] - min_v[0], max_v[1] - min_v[1], max_v[2] - min_v[2] };
        double shift[3] = { - min_v[0] - crange[0]/2, - min_v[1] - crange[1]/2, - min_v[2] - crange[2]/2 };

        if (TOL > crange[0] || TOL > crange[1] || TOL > crange[2])
                error("coordinate range of stl in one dimension is almost zero");

        double m0 = (crange[0] > crange[1]) ? crange[0] : crange[1];
        double m = (crange[2] > m0) ? crange[2] : m0;
        double scale[3] = { fov_size/m, fov_size/m, fov_size/m };

        stl_shift_model(dims, model, shift);
        stl_scale_model(dims, model, scale);
}

void stl_stats(const long dims[3], const double* model)
{
        debug_printf(DP_INFO, "Number of triangles: %ld\n", dims[2]);
        double min_v[3], max_v[3];
        stl_coordinate_limits(dims, model, min_v, max_v);
        debug_printf(DP_INFO, "Vertex coordinate ranges:\n");
        debug_printf(DP_INFO, "1: (%lf,%lf)\n", min_v[0], max_v[0]);
        debug_printf(DP_INFO, "2: (%lf,%lf)\n", min_v[1], max_v[1]);
        debug_printf(DP_INFO, "3: (%lf,%lf)\n", min_v[2], max_v[2]);
}

void stl_print(const long dims[3], const double* model)
{
        debug_printf(DP_INFO, "Number of triangles: %ld\n", dims[2]);
        long strs[3], pos[3];
        md_calc_strides(3, strs, dims, DL_SIZE);
        md_set_dims(3, pos, 0);

        for (pos[2] = 0; pos[2] < dims[2]; pos[2]++) {

                debug_printf(DP_INFO, "Triangle: %ld\n", pos[2]);

                double v0[3], v1[3], v2[3], n[3];
                pos[1] = 0;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        v0[pos[0]] = MD_ACCESS(3, strs, pos, model);

                pos[1] = 1;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        v1[pos[0]] = MD_ACCESS(3, strs, pos, model);

                pos[1] = 2;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        v2[pos[0]] = MD_ACCESS(3, strs, pos, model);

                pos[1] = 3;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        n[pos[0]] = MD_ACCESS(3, strs, pos, model);

                debug_printf(DP_INFO, "V0: %f %f %f\n", v0[0], v0[1], v0[2]);
                debug_printf(DP_INFO, "V0: %f %f %f\n", v1[0], v1[1], v1[2]);
                debug_printf(DP_INFO, "V0: %f %f %f\n", v1[0], v2[1], v2[2]);
                debug_printf(DP_INFO, "N:  %f %f %f\n", n[0], n[1], n[2]);
        }
}

void stl_compute_normals(const long dims[3], double* model)
{
        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos0[3] = { };
		long pos1[3] = { };
		long pos2[3] = { };
		long posn[3] = { };

                pos0[2] = i;
                pos1[2] = i;
                pos2[2] = i;
                posn[2] = i;
                pos1[1] = 1;
                pos2[1] = 2;
                posn[1] = 3;

                double d1[3], d2[3];

                vec3d_saxpy(d1, &MD_ACCESS(3, strs, pos0, model), -1, &MD_ACCESS(3, strs, pos1, model));
                vec3d_saxpy(d2, &MD_ACCESS(3, strs, pos0, model), -1, &MD_ACCESS(3, strs, pos2, model));

		double nt[3];
		vec3d_cp(nt, d1, d2);
		vec3d_saxpy(&MD_ACCESS(3, strs, posn, model), nt, 1 / vec3d_norm(nt), NULL);
        }
}

// stl_str_ {C}ontained {F}irst {I}ndex
// returns the first index at which s0 is contained in s1
static int stl_str_cfi(const char* s0, const char* s1)
{
        int l0 = strlen(s0);
        int l1 = strlen(s1);

        if (l0 > l1)
                return -1;

        for (int i = 0; i < l1 - l0; i++)
                if (0 == strncmp(s0, &s1[i], (size_t) l0))
                        return i;

        return -1;
}

// call stl_str_cfi but with error instead of -1 return
static int stl_str_cfie(const char* s0, const char* s1)
{
        int r = stl_str_cfi(s0, s1);

        if (-1 == r)
                error("String '%s' not contained in %s\n", s0, s1);

        return r;
}

// check if s0 is contained anywhere in s1
static bool stl_str_contained(const char* s0, const char* s1)
{
        return -1 != stl_str_cfi(s0, s1);
}

void stl_write_binary(const char* name, const long dims[3], const double* model)
{
        const size_t hs = 80 + sizeof(int32_t);
        const size_t bs = 12 * FL_SIZE + sizeof(uint16_t);
        const size_t d2 = (size_t) dims[2];
        const size_t s = hs + d2 * bs;

        // write header
        char* buf = xmalloc(s);

        for (int i = 0; i < (int) s; i++)
                buf[(size_t) i] = ' ';

        strcpy(buf, "Created by bart ");
        strcat(buf, bart_version);

        *(uint32_t*) &buf[80] = (uint32_t) dims[2];

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos[3] = { };
                pos[2] = i;
                pos[1] = 3;

                for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                        float f = MD_ACCESS(3, strs, pos, model);
                        memcpy(&buf[hs + (size_t) i * bs + (size_t) pos[0] * FL_SIZE], &f, FL_SIZE);
                }

                for (pos[1] = 0; pos[1] < 3; pos[1]++) {

                        for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                                float f = MD_ACCESS(3, strs, pos, model);
                                memcpy(&buf[hs + (size_t) i * bs + (size_t) (3 + pos[0] + 3 * pos[1]) * FL_SIZE], &f, FL_SIZE);
                        }
                }
        }

        int fd = open(name, O_WRONLY | O_CREAT, 0666);

        if (-1 == fd)
                error("write stl error fd\n", name);

        if ((int) s != xwrite(fd, s, buf))
                error("write stl error %s\n", name);

        close(fd);
        xfree(buf);
}

// check encoding of file
static bool stl_is_ascii(const char* name)
{
        int fd = open(name, O_RDONLY);

        if (-1 == fd)
                error("is ascii read stl error %s\n", name);

        char tmp[80];

        if (80 != xread(fd, 80, tmp))
                error("stl file could not be opened\n");

        close(fd);

        int c = 0;

        for (int i = 0; i < 80; i++)
                if ('\n' == tmp[i]) {

                        c = i;
                        break;
                }

        tmp[c] = '\0';

        return stl_str_contained("solid", tmp);
}


#define MAX_LINE_LENGTH 128

static double* stl_read_ascii(const char* name, long dims[3])
{
        FILE* fp = fopen(name, "r");

        if (NULL == fp)
                error("read stl error %s\n", name);

        const char fn[] = "facet normal";
        const char ve[] = "vertex";

        char line[MAX_LINE_LENGTH];
        int N = 0;

        while (NULL != fgets(line, sizeof line, fp))
                if (stl_str_contained(fn, line))
                        N++;

        fclose(fp);

        dims[0] = 3;
        dims[1] = 4;
        dims[2] = N;

        double* model = md_alloc(3, dims, DL_SIZE);

        char* l0 = NULL;
        char* l1 = NULL;
        char* l2 = NULL;
        char* l3 = NULL;
        char* l4 = NULL;
        char* l5 = NULL;
        char* l6 = NULL;
        int p;
        int r;

        // start again with the beginning of file
        fp = fopen(name, "r");

        if (NULL == fp)
                error("read stl error %s\n", name);

        // skip the first line
	if (NULL == fgets(line, sizeof line, fp))
		error("error reading stl file\n");

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

	long pos[3] = { };

        // ASCII encoded stl files have the following repeating pattern:
        // facet normal FLOAT FLOAT FLOAT
        // outer loop
        // vertex FLOAT FLOAT FLOAT
        // vertex FLOAT FLOAT FLOAT
        // vertex FLOAT FLOAT FLOAT
        // endloop
        // endfacet
        //
        // the last line of the file will contain:
        // endsolid
        float f[3];

        for (int n = 0; n < N; n++) {

                pos[2] = n;
                // facet normal
		if (NULL == (l0 = fgets(line, sizeof line, fp)))
			error("error reading stl file\n");

                // index at which string starts
                p = stl_str_cfie(fn, l0);

                if (3 != (r = sscanf(&l0[p + (int)strlen(fn)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", fn, n);

                pos[1] = 3;

                for (int i = 0; i < 3; i++) {

                        pos[0] = i;
                        MD_ACCESS(3, strs, pos, model) = (double) f[i];
                }

                l0 = NULL;

		if (NULL == (l1 = fgets(line, sizeof line, fp)))
			error("error reading stl file\n");

                stl_str_cfie("outer loop", l1);
                l1 = NULL;

		if (NULL == (l2 = fgets(line, sizeof line, fp)))
			error("error reading stl file\n");

                p = stl_str_cfie(ve, l2);

                if (3 != (r = sscanf(&l2[p + (int)strlen(ve)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", ve, n);

                pos[1] = 0;

                for (int i = 0; i < 3; i++) {

                        pos[0] = i;
                        MD_ACCESS(3, strs, pos, model) = (double) f[i];
                }

                l2 = NULL;

		if (NULL == (l3 = fgets(line, sizeof line, fp)))
			error("error reading stl file\n");

                p = stl_str_cfie(ve, l3);

                if (3 != (r = sscanf(&l3[p + (int)strlen(ve)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", ve, n);

                pos[1] = 1;

                for (pos[0]= 0; pos[0] < 3; pos[0]++)
                        MD_ACCESS(3, strs, pos, model) = (double) f[pos[0]];

                l3 = NULL;

		if (NULL == (l4 = fgets(line, sizeof line, fp)))
			error("error reading stl file\n");

                p = stl_str_cfie(ve, l4);

                if (3 != (r = sscanf(&l4[p + (int)strlen(ve)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", ve, n);

                pos[1] = 2;

                for (pos[0] = 0; pos[0] < 3; pos[0]++)
                        MD_ACCESS(3, strs, pos, model) = (double) f[pos[0]];

                l4 = NULL;

		if (NULL == (l5 = fgets(line, sizeof line, fp)))
			error("error reading stl file\n");

                stl_str_cfie("endloop", l5);
                l5 = NULL;

		if (NULL == (l6 = fgets(line, sizeof line, fp)))
			error("error reading stl file\n");

                stl_str_cfie("endfacet", l6);
                l6 = NULL;
        }

	if (NULL == (l0 = fgets(line, sizeof line, fp)))
		error("error reading stl file\n");

        stl_str_cfie("endsolid", l0);

        fclose(fp);

        return model;
}



/**
 * order of data in binary encoded files (www.fabbers.com/tech/STL_Format#Sct_binary)
 *
 * header
 * 80 Bytes (not relevant)
 * 4 Byte (N = number of blocks)
 *
 * N * block composed of
 * 3 x 4 Byte (Normal vector coordinates)
 * 3 x 4 Byte (vertex 1 coordinates)
 * 3 x 4 Byte (vertex 2 coordinates)
 * 3 x 4 Byte (vertex 3 coordinates)
 * 1 x 2 Byte (not relevant)
 *
 * total file size: 84 + N * (12 * 4 + 2) Byte
 * block size: 50 Byte
 **/
// read binary encoded stl files.
static double* stl_read_binary(const char* name, long dims[3])
{
        int fd = open(name, O_RDONLY);

        if (-1 == fd)
                error("read stl error open %s\n", name);

        char tmp[80];

        if (80 != xread(fd, 80, tmp))
                error("stl file could not be opened\n");

        uint32_t Nu;

        if (sizeof(uint32_t) != xread(fd, sizeof(uint32_t), (char* ) &Nu))
                error("stl file could not be opened\n");

        int N = (int) Nu;

        dims[0] = 3;
        dims[1] = 4;
        dims[2] = N;

        double* model = md_calloc(3, dims, DL_SIZE);

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);


        const size_t bs = 12 * FL_SIZE + sizeof(uint16_t);
        const size_t L = (size_t) N * bs;
        char* buf = xmalloc(L);

        if ((int) L != xread(fd, L, buf))
                error("stl file could not be opened\n");

        close(fd);

#pragma omp parallel for
        for (int i = 0; i < N; i++) {

                long pos[3];
                md_set_dims(3, pos, 0);
                pos[2] = i;
                pos[1] = 3;

                for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                        float f;
                        memcpy(&f, &buf[(size_t) i * bs + (size_t) pos[0] * FL_SIZE], FL_SIZE);
                        MD_ACCESS(3, strs, pos, model) = f;
                }

                for (pos[1] = 0; pos[1] < 3; pos[1]++) {

                        for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                                float f;
                                memcpy(&f, &buf[(size_t) i * bs + (size_t) (3 + pos[0] + 3 * pos[1]) * FL_SIZE], FL_SIZE);
                                MD_ACCESS(3, strs, pos, model) = f;
                        }
                }
        }
        xfree(buf);
        return model;
}

double* stl_read(const char *name, long dims[3])
{
        return (stl_is_ascii(name) ? stl_read_ascii : stl_read_binary)(name, dims);
}

bool stl_fileextension(const char* name)
{
	char* ext = strrchr(name, '.');

        if (NULL != ext)
		if (0 == strcmp(ext, ".stl"))
                        return true;

        return false;
}

// convert model in cfl md array to model in double md array
double* stl_cfl2d(const long dims[3], const complex float* cmodel)
{
        double* model = md_alloc(3, dims, DL_SIZE);

        long pos[3] = { };

	long dstrs[3];
        md_calc_strides(3, dstrs, dims, DL_SIZE);

	long cstrs[3];
        md_calc_strides(3, cstrs, dims, CFL_SIZE);

        do {
                MD_ACCESS(3, dstrs, pos, model) = (float) MD_ACCESS(3, cstrs, pos, cmodel);

        } while (md_next(3, dims, ~0UL, pos));

        return model;
}

// convert model in double md array to model in cfl md array
void stl_d2cfl(const long dims[3], complex float* cmodel, const double* model)
{
	long pos[3] = { };

        long dstrs[3];
        md_calc_strides(3, dstrs, dims, DL_SIZE);

        long cstrs[3];
        md_calc_strides(3, cstrs, dims, CFL_SIZE);

        do {

                MD_ACCESS(3, cstrs, pos, cmodel) = (float) MD_ACCESS(3, dstrs, pos, model) + 0.j;

        } while (md_next(3, dims, ~0UL, pos));
}

// compute relative position (shift, rotation, ...) of the triangle wrt to the origin and z-axis
void stl_relative_position(struct triangle* t)
{
	vec3d_saxpy(t->e0, t->v0, -1., t->v1);
	vec3d_saxpy(t->e1, t->v0, -1., t->v2);

	assert(0. < vec3d_norm(t->e0));
	assert(0. < vec3d_norm(t->e1));

	double sn[3];
	vec3d_cp(sn, t->e0, t->e1);
	t->sur = 0.5 * vec3d_norm(sn);

	// compute b0, b1 as orthogonal basis vectors of the plane which contains the triangle
	double b0[3], tmp[3], b1[3];
	vec3d_saxpy(b0, t->e0, 1 / vec3d_norm(t->e0), NULL);
	vec3d_saxpy(tmp, t->e1, 1 / vec3d_norm(t->e1), NULL);

	// b1 is orthogonal component of tmp wrt b0
	double f = -1 * vec3d_sdot(b0, tmp) / vec3d_norm(b0);
	vec3d_saxpy(b1, b0, f, tmp);
	vec3d_saxpy(b1, b1, 1 / vec3d_norm(b1), NULL);

	// compute angle between normal vector and z axis
	double ez[3] = { 0., 0., 1. };
	t->angle = vec3d_angle(ez, t->n);

	// if normal vector is -ez
	if (1E-10 > fabs(M_PI - t->angle) || 1E-10 > fabs(t->angle)) {

		t->rot[0] = 1.;
		t->rot[1] = 0.;
		t->rot[2] = 0.;

	} else {

		vec3d_cp(t->rot, t->n, ez);
	}

	vec3d_saxpy(t->rot, t->rot, 1. / vec3d_norm(t->rot), NULL);

	// compute center of triangle
	vec3d_set(t->ctr, 0);
	vec3d_saxpy(t->ctr, t->v0, 1. / 3., t->ctr);
	vec3d_saxpy(t->ctr, t->v1, 1. / 3., t->ctr);
	vec3d_saxpy(t->ctr, t->v2, 1. / 3., t->ctr);

	// compute centered triangle
	double v0c[3], v1c[3], v2c[3];
	vec3d_saxpy(v0c, t->ctr, -1, t->v0);
	vec3d_saxpy(v1c, t->ctr, -1, t->v1);
	vec3d_saxpy(v2c, t->ctr, -1, t->v2);

	// compute centered rotated triangle
	double v0cr[3], v1cr[3], v2cr[3];
	vec3d_rotax(v0cr, t->angle, t->rot, v0c);
	vec3d_rotax(v1cr, t->angle, t->rot, v1c);
	vec3d_rotax(v2cr, t->angle, t->rot, v2c);

	t->poly[0] = v0cr[0];
	t->poly[1] = v0cr[1];
	t->poly[2] = v1cr[0];
	t->poly[3] = v1cr[1];
	t->poly[4] = v2cr[0];
	t->poly[5] = v2cr[1];

	// signed volume of tetrahedron triangle + origin
	vec3d_cp(tmp, t->v0, t->v1);
	t->svol = copysign(vec3d_sdot(tmp, t->v2) / 6, vec3d_sdot(t->v0, t->n));
}

struct triangle_stack* stl_preprocess_model(const long dims[3], const double* model)
{
	struct triangle_stack* ts = xmalloc(sizeof(struct triangle_stack));

	ts->N = dims[2];
	ts->tri = xmalloc((size_t) ts->N * sizeof(struct triangle));

	long tdims[3];
	md_copy_dims(3, tdims, dims);

	long tstrs[3];
	md_calc_strides(3, tstrs, tdims, DL_SIZE);

#pragma omp parallel for
	for (int i = 0; i < ts->N; i++) {

		long pos[3] = { [2] = i };

		memcpy(&ts->tri[i], &MD_ACCESS(3, tstrs, pos, model), 12 * DL_SIZE);

		stl_relative_position(&ts->tri[i]);
	}
	return ts;
}

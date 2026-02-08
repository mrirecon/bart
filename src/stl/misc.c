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
#include <limits.h>
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
static void stl_coordinate_limits(const long dims[3], const double* model, double min_v[3], double max_v[3])
{
	assert(3 == dims[1]);

	for (int i = 0; i < 3; i++) {

		min_v[i] = +INFINITY;
		max_v[i] = -INFINITY;
	}

	long strs[3];
	md_calc_strides(3, strs, dims, DL_SIZE);

	for (int j = 0; j < dims[0]; j++) {
		for (int k = 0; k < dims[1]; j++) {
			for (int l = 0; l < dims[2]; l++) {

				long pos[3] = { j, k, l };
				double val = MD_ACCESS(3, strs, pos, model);

				min_v[j] = MIN(min_v[j], val);
				max_v[j] = MAX(max_v[j], val);
			}
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

                long pos[3] = { [2] = i };

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

                long pos[3] = { [2] = i };

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        for (pos[1] = 0; pos[1] < dims[1] - 1; pos[1]++)
                                MD_ACCESS(3, strs, pos, model) += shift[pos[0]];
        }
}

// shift and scale the model to FOV of size fov_size > 0.
void stl_center_fov(const long dims[3], double* model, double fov_size)
{
        if (0. >= fov_size)
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

        double min_v[3];
	double max_v[3];

        stl_coordinate_limits(dims, model, min_v, max_v);

        debug_printf(DP_INFO, "Vertex coordinate ranges:\n");
        debug_printf(DP_INFO, "1: (%lf,%lf)\n", min_v[0], max_v[0]);
        debug_printf(DP_INFO, "2: (%lf,%lf)\n", min_v[1], max_v[1]);
        debug_printf(DP_INFO, "3: (%lf,%lf)\n", min_v[2], max_v[2]);
}

void stl_print(const long dims[3], const double* model)
{
	assert(3 == dims[0]);
	assert(4 == dims[1]);

        debug_printf(DP_INFO, "Number of triangles: %ld\n", dims[2]);

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

	long pos[3] = { };

        for (pos[2] = 0; pos[2] < dims[2]; pos[2]++) {

                debug_printf(DP_INFO, "Triangle: %ld\n", pos[2]);

		const char* str[] = { "v0", "v1", "v2", "n" };

		for (pos[1] = 0; pos[1] < 4; pos[1]++) {

			double v[3];
			for (pos[0] = 0; pos[0] < 3; pos[0]++)
				v[pos[0]] = MD_ACCESS(3, strs, pos, model);

			debug_printf(DP_INFO, "%s: %f %f %f\n", str[pos[1]], v[0], v[1], v[2]);
		}
        }
}

void stl_compute_normals(const long dims[3], double* model)
{
        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos0[3] = { [1] = 0, [2] = i };
		long pos1[3] = { [1] = 1, [2] = i };
		long pos2[3] = { [1] = 2, [2] = i };
		long posn[3] = { [1] = 3, [2] = i };

                double d1[3];
		double d2[3];

                vec3d_saxpy(d1, &MD_ACCESS(3, strs, pos0, model), -1., &MD_ACCESS(3, strs, pos1, model));
                vec3d_saxpy(d2, &MD_ACCESS(3, strs, pos0, model), -1., &MD_ACCESS(3, strs, pos2, model));

		double nt[3];
		vec3d_crossproduct(nt, d1, d2);
		vec3d_saxpy(&MD_ACCESS(3, strs, posn, model), nt, 1. / vec3d_norm(nt), NULL);
        }
}



enum { TRI_SIZE = 12 * FL_SIZE + (int)sizeof(uint16_t) };

struct stl_triangle {

	_Float32 nv[3];
	_Float32 v[3][3];
	uint16_t abc;	// attribute byte count
};

void stl_write_binary(const char* name, const long dims[3], const double* model)
{
	int fd = open(name, O_WRONLY | O_CREAT, 0666);

        if (-1 == fd)
                error("opening stl file for writing\n", name);


	// FIXME: little endian
        char header[80 + (int)sizeof(int32_t)];
	memset(header, 0, sizeof(header));
        snprintf(header, 80, "Created by BART %s.\n", bart_version);
	memcpy(&header[80], &(uint32_t){ (uint32_t)dims[2] }, sizeof(uint32_t));

	if (sizeof(header) != xwrite(fd, sizeof(header), header))
                error("write stl error %s\n", name);

	// write triangles

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

        for (int i = 0; i < dims[2]; i++) {

		struct stl_triangle tri = { };
		static_assert(TRI_SIZE <= sizeof(tri));

                long pos[3] = { [1] = 3, [2] = i };

                for (int k = 0; k < 3; k++)
                        tri.nv[k] = MD_ACCESS(3, strs, (pos[0] = k, pos), model);

                for (int j = 0; j < 3; j++) {

			pos[1] = j;

			for (int k = 0; k < 3; k++)
                                tri.v[j][k] = MD_ACCESS(3, strs, (pos[0] = k, pos), model);
		}

		if (TRI_SIZE != xwrite(fd, TRI_SIZE, (void*)&tri))
			error("write stl error %s\n", name);
        }

        close(fd);
}



#define MAX_LINE_LENGTH 128


static void stl_read_ascii(const char* name, long dims[3], double* model)
{
        FILE* fp = fopen(name, "r");

        if (NULL == fp)
                error("read stl error %s\n", name);

        long strs[3];

	if (NULL != model)
		md_calc_strides(3, strs, dims, DL_SIZE);

	long pos[3] = { };
        char line[MAX_LINE_LENGTH];

	NESTED(bool, keyword, (const char* kw))
	{
		int end = 0;
		return 0 == sscanf(line, kw, &end) && '\0' == line[end];
	}

	NESTED(bool, keyword_args, (const char* kw))
	{
		int end = 0;
		float f[3];

		if (3 != sscanf(line, kw, &f[0], &f[1], &f[2], &end) || '\0' != line[end])
			return false;

		if (NULL == model)
			return true;

		for (int k = 0; k < 3; k++)
			MD_ACCESS(3, strs, (pos[0] = k, pos), model) = f[k];

		return true;
	};

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

	if (NULL == fgets(line, sizeof line, fp))
		error("error reading stl file\n");

	if (!(keyword("solid %*s\n%n") || keyword("solid\n%n")))

		fclose(fp);
		return;
	}

	int n = 0;

        while (true) {

		if (NULL != model && dims[2] == pos[2])
			break;

                pos[2] = n++;

		if (NULL == fgets(line, sizeof line, fp))
			error("error reading stl file\n");

                pos[1] = 3;
		if (!keyword_args(" facet normal %f %f %f\n%n"))
			break;

		if (NULL == fgets(line, sizeof line, fp))
			error("error reading stl file\n");

		if (!keyword(" outer loop\n%n"))
			error("error reading stl file\n");

		for (int i = 0; i < 3; i++) {

			pos[1] = i;

			if (NULL == fgets(line, sizeof line, fp))
				error("error reading stl file\n");

			if (!keyword_args(" vertex %f %f %f\n%n"))
				error("error reading stl file\n");
		}

		if (NULL == fgets(line, sizeof line, fp))
			error("error reading stl file\n");

		if (!keyword(" endloop\n%n"))
			error("error reading stl file\n");

		if (NULL == fgets(line, sizeof line, fp))
			error("error reading stl file\n");

		if (!keyword(" endfacet\n%n"))
			error("error reading stl file\n");
        }

	if (!(keyword(" endsolid %*s\n%n") || keyword(" endsolid\n%n")))
		error("error reading stl file\n");

	if (NULL == model) {

		dims[0] = 3;
		dims[1] = 4;
		dims[2] = pos[2];
	}

        fclose(fp);
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
                error("read stl error open %s.", name);

        char tmp[80];

        if (80 != xread(fd, 80, tmp))
                error("stl file could not be read.");

        uint32_t Nu;

        if (sizeof(uint32_t) != xread(fd, sizeof(uint32_t), (char* )&Nu))
                error("stl file could not be read.");

	if (INT_MAX < Nu)
		error("too many triangles.");

        int N = (int)Nu;

        dims[0] = 3;
        dims[1] = 4;
        dims[2] = N;

        double* model = md_calloc(3, dims, DL_SIZE);

        long strs[3];
        md_calc_strides(3, strs, dims, DL_SIZE);

        for (int i = 0; i < N; i++) {

		struct stl_triangle tri = { };
		static_assert(TRI_SIZE <= sizeof(tri));

		if (TRI_SIZE != xread(fd, TRI_SIZE, (char*)&tri))
			error("stl file could not be read\n");

                long pos[3] = { [2] = i };
                pos[1] = 3;

                for (int k = 0; k < 3; k++)
                        MD_ACCESS(3, strs, (pos[0] = k, pos), model) = tri.nv[k];

                for (int j = 0; j < 3; j++) {

			pos[1] = j;

			for (int k = 0; k < 3; k++)
                                MD_ACCESS(3, strs, (pos[0] = k, pos), model) = tri.v[j][k];
		}
        }

        close(fd);

        return model;
}

double* stl_read(const char *name, long dims[3])
{
	dims[2] = 0;
	stl_read_ascii(name, dims, NULL);

	if (0 < dims[2]) {

		double* model = md_alloc(3, dims, DL_SIZE);
		stl_read_ascii(name, dims, model);
		return model;
	}

        return stl_read_binary(name, dims);
}

bool stl_fileextension(const char* name)
{
	char* ext = strrchr(name, '.');

        if (NULL == ext)
		return false;

	return (0 == strcmp(ext, ".stl"));
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
	vec3d_crossproduct(sn, t->e0, t->e1);
	t->sur = 0.5 * vec3d_norm(sn);

	// compute b0, b1 as orthogonal basis vectors of the plane which contains the triangle
	double b0[3], tmp[3], b1[3];
	vec3d_saxpy(b0, t->e0, 1. / vec3d_norm(t->e0), NULL);
	vec3d_saxpy(tmp, t->e1, 1. / vec3d_norm(t->e1), NULL);

	// b1 is orthogonal component of tmp wrt b0
	double f = -1. * vec3d_sdot(b0, tmp) / vec3d_norm(b0);
	vec3d_saxpy(b1, b0, f, tmp);
	vec3d_saxpy(b1, b1, 1. / vec3d_norm(b1), NULL);

	// compute angle between normal vector and z axis
	double ez[3] = { 0., 0., 1. };
	t->angle = vec3d_angle(ez, t->n);

	// if normal vector is -ez
	if (1E-10 > fabs(M_PI - t->angle) || 1E-10 > fabs(t->angle)) {

		t->rot[0] = 1.;
		t->rot[1] = 0.;
		t->rot[2] = 0.;

	} else {

		vec3d_crossproduct(t->rot, t->n, ez);
	}

	vec3d_saxpy(t->rot, t->rot, 1. / vec3d_norm(t->rot), NULL);

	// compute center of triangle
	vec3d_set(t->ctr, 0);
	vec3d_saxpy(t->ctr, t->v0, 1. / 3., t->ctr);
	vec3d_saxpy(t->ctr, t->v1, 1. / 3., t->ctr);
	vec3d_saxpy(t->ctr, t->v2, 1. / 3., t->ctr);

	// compute centered triangle
	double v0c[3], v1c[3], v2c[3];
	vec3d_saxpy(v0c, t->ctr, -1., t->v0);
	vec3d_saxpy(v1c, t->ctr, -1., t->v1);
	vec3d_saxpy(v2c, t->ctr, -1., t->v2);

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
	vec3d_crossproduct(tmp, t->v0, t->v1);
	t->svol = copysign(vec3d_sdot(tmp, t->v2) / 6., vec3d_sdot(t->v0, t->n));
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


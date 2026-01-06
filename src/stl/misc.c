/* Copyright 2025. University Medical Center Göttingen, Germany
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
#include "num/flpmath.h"

#include "misc/io.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/version.h"

#include "stl/misc.h"

void print_vec(int N, const double* d)
{
        for (int i = 0; i < N; i++)
                debug_printf(DP_INFO, "%lf ", d[i]);

        debug_printf(DP_INFO, "\n");
}

// o = v2 - v1
void stl_sub_vec3(double* o, const double* v1, const double* v2) 
{
        for (int i = 0; i < 3; i++)
                o[i] = v2[i] - v1[i];
}

static double stl_ip_vec(int N, const double* d0, const double* d1)
{
        double l = 0;

        for (int i = 0; i < N; i++)
                l += d0[i] * d1[i];
        
        return l;
}

double stl_norm_vec(int N, const double* d)
{
        return sqrt(stl_ip_vec(N, d, d));
}

// crossproduct
static void stl_cp(double* o, const double* v0, const double* v1)
{
        o[0] = v0[1] * v1[2] - v0[2] * v1[1];
        o[1] = v0[2] * v1[0] - v0[0] * v1[2];
        o[2] = v0[0] * v1[1] - v0[1] * v1[0];
}

// unit normal vector right hand rule
void stl_unormal_vec3(double* n, const double* v0, const double* v1)
{
        double nt[3];

        stl_cp(nt, v0, v1);
        double l = stl_norm_vec(3, nt);

        if (TOL > l)
                error("Vector length is zero and can not be normalized.\n");

        n[0] = nt[0] / l;
        n[1] = nt[1] / l;
        n[2] = nt[2] / l;
}

// compute minimal and maximal vertex coordinates
static void stl_coordinate_limits(int D, long dims[D], double* model, double* min_v, double* max_v)
{
        min_v[0] = INFINITY;
        min_v[1] = INFINITY;
        min_v[2] = INFINITY;
        max_v[0] = -INFINITY;
        max_v[1] = -INFINITY;
        max_v[2] = -INFINITY;

        long strs[D];
        md_calc_strides(D, strs, dims, D_SIZE);
        long pos[D];
        md_set_dims(D, pos, 0);

        long pos0[D], pos1[D], pos2[D];
        md_set_dims(D, pos0, 0);
        md_set_dims(D, pos1, 0);
        md_set_dims(D, pos2, 0);
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

                        double v0 = MD_ACCESS(D, strs, pos0, model);
                        double v1 = MD_ACCESS(D, strs, pos1, model);
                        double v2 = MD_ACCESS(D, strs, pos2, model);
                       
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
void stl_scale_model(int D, long dims[D], double* model, double scale[3])
{
        long strs[D];
        md_calc_strides(D, strs, dims, D_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos[D];
                md_set_dims(D, pos, 0);
                pos[2] = i;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        for (pos[1] = 0; pos[1] < dims[1] - 1; pos[1]++)
                                MD_ACCESS(D, strs, pos, model) *= scale[pos[0]];
        }
}

// Shifts all vertex coordinates by shift vector. It doesn't shift the normal vector (shift invariant)
void stl_shift_model(int D, long dims[D], double* model, double shift[3])
{
        long strs[D];
        md_calc_strides(D, strs, dims, D_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos[D];
                md_set_dims(D, pos, 0);
                pos[2] = i;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        for (pos[1] = 0; pos[1] < dims[1] - 1; pos[1]++)
                                MD_ACCESS(D, strs, pos, model) += shift[pos[0]];
        }
}

// shift and scale the model to FOV of size fov_size > 0.
void stl_center_fov(int D, long dims[D], double* model, double fov_size)
{
        if (0 >= fov_size)
                error("fov_size should be positive.");

        double min_v[3], max_v[3];
        stl_coordinate_limits(D, dims, model, min_v, max_v);
        double crange[3] = { max_v[0] - min_v[0], max_v[1] - min_v[1], max_v[2] - min_v[2] };
        double shift[3] = { - min_v[0] - crange[0]/2, - min_v[1] - crange[1]/2, - min_v[2] - crange[2]/2 };

        if (TOL > crange[0] || TOL > crange[1] || TOL > crange[2])
                error("coordinate range of stl in one dimension is almost zero");

        double m0 = (crange[0] > crange[1]) ? crange[0] : crange[1];
        double m = (crange[2] > m0) ? crange[2] : m0;
        double scale[3] = { fov_size/m, fov_size/m, fov_size/m };

        stl_shift_model(D, dims, model, shift);
        stl_scale_model(D, dims, model, scale);
}

void stl_stats(int D, long dims[D], double* model)
{
        debug_printf(DP_INFO, "Number of triangles: %ld\n", dims[2]);
        double min_v[3], max_v[3];
        stl_coordinate_limits(D, dims, model, min_v, max_v);
        debug_printf(DP_INFO, "Vertex coordinate ranges:\n");
        debug_printf(DP_INFO, "1: (%lf,%lf)\n", min_v[0], max_v[0]);
        debug_printf(DP_INFO, "2: (%lf,%lf)\n", min_v[1], max_v[1]);
        debug_printf(DP_INFO, "3: (%lf,%lf)\n", min_v[2], max_v[2]);
}

void stl_print(int D, long dims[D], double* model)
{
        debug_printf(DP_INFO, "Number of triangles: %ld\n", dims[2]);
        long strs[D], pos[D];
        md_calc_strides(D, strs, dims, D_SIZE);
        md_set_dims(D, pos, 0);

        for (pos[2] = 0; pos[2] < dims[2]; pos[2]++) {
                
                debug_printf(DP_INFO, "Triangle: %ld\n", pos[2]);

                double v0[3], v1[3], v2[3], n[3];
                pos[1] = 0;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        v0[pos[0]] = MD_ACCESS(D, strs, pos, model);

                pos[1] = 1;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        v1[pos[0]] = MD_ACCESS(D, strs, pos, model);

                pos[1] = 2;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        v2[pos[0]] = MD_ACCESS(D, strs, pos, model);

                pos[1] = 3;

                for (pos[0] = 0; pos[0] < dims[0]; pos[0]++)
                        n[pos[0]] = MD_ACCESS(D, strs, pos, model);

                debug_printf(DP_INFO, "V0:");
                print_vec(3, v0);
                debug_printf(DP_INFO, "V1:");
                print_vec(3, v1);
                debug_printf(DP_INFO, "V2:");
                print_vec(3, v2);
                debug_printf(DP_INFO, "N:");
                print_vec(3, n);
        }
}

void stl_compute_normals(int D, long dims[D], double* model)
{
        long strs[D];
        md_calc_strides(D, strs, dims, D_SIZE);

#pragma omp parallel for
        for (int i = 0; i < dims[2]; i++) {

                long pos0[D], pos1[D], pos2[D], posn[D];
                md_set_dims(D, pos0, 0);
                md_set_dims(D, pos1, 0);
                md_set_dims(D, pos2, 0);
                md_set_dims(D, posn, 0);
                pos0[2] = i;
                pos1[2] = i;
                pos2[2] = i;
                posn[2] = i;
                pos1[1] = 1;
                pos2[1] = 2;
                posn[1] = 3;

                double d1[3], d2[3];

                stl_sub_vec3(d1, &MD_ACCESS(D, strs, pos0, model), &MD_ACCESS(D, strs, pos1, model));
                stl_sub_vec3(d2, &MD_ACCESS(D, strs, pos0, model), &MD_ACCESS(D, strs, pos2, model));
                stl_unormal_vec3(&MD_ACCESS(D, strs, posn, model), d1, d2);
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

void stl_write_binary(int D, long dims[D], double* model, const char* name)
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

        long strs[D];
        md_calc_strides(D, strs, dims, D_SIZE);

#pragma omp parallel for        
        for (int i = 0; i < dims[2]; i++) {

                long pos[D];
                md_set_dims(D, pos, 0);
                pos[2] = i;
                pos[1] = 3;

                for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                        float f = MD_ACCESS(D, strs, pos, model);
                        memcpy(&buf[hs + (size_t) i * bs + (size_t) pos[0] * FL_SIZE], &f, FL_SIZE);
                }

                for (pos[1] = 0; pos[1] < 3; pos[1]++) {

                        for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                                float f = MD_ACCESS(D, strs, pos, model);
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

static bool stl_read_line(char** line, FILE* s)
{
        size_t l;
        ssize_t r;

        if (-1 == (r = getline(line, &l, s)))
                return false;

        return true;
}

static double* stl_read_ascii(int D, long dims[D], const char* name)
{
        FILE* ptr = fopen(name, "r");

        if (NULL == ptr)
                error("read stl error %s\n", name);
        
        const char fn[] = "facet normal";
        const char ve[] = "vertex";

        char* line = NULL;
        int N = 0;

        while (stl_read_line(&line, ptr)) {

                if (stl_str_contained(fn, line))
                        N++;

                xfree(line);
                line = NULL;
        }

        xfree(line);
        fclose(ptr);

        md_set_dims(D, dims, 1);
        dims[0] = 3;
        dims[1] = 4;
        dims[2] = N;
        double* model = md_alloc(D, dims, D_SIZE);
        
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
        ptr = fopen(name, "r");

        if (NULL == ptr)
                error("read stl error %s\n", name);

        // skip the first line
        stl_read_line(&l0, ptr);
        xfree(l0);
        l0 = NULL;

        long strs[D], pos[D];
        md_calc_strides(D, strs, dims, D_SIZE);
        md_set_dims(D, pos, 0);
        
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
                stl_read_line(&l0, ptr);
                // index at which string starts
                p = stl_str_cfie(fn, l0);

                if (3 != (r = sscanf(&l0[p + (int)strlen(fn)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", fn, n);

                pos[1] = 3;

                for (int i = 0; i < 3; i++) {

                        pos[0] = i;
                        MD_ACCESS(D, strs, pos, model) = (double) f[i];
                }

                xfree(l0);
                l0 = NULL;

                stl_read_line(&l1, ptr);
                stl_str_cfie("outer loop", l1);
                xfree(l1);
                l1 = NULL;
                
                stl_read_line(&l2, ptr);
                p = stl_str_cfie(ve, l2);

                if (3 != (r = sscanf(&l2[p + (int)strlen(ve)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", ve, n);

                pos[1] = 0;

                for (int i = 0; i < 3; i++) {

                        pos[0] = i;
                        MD_ACCESS(D, strs, pos, model) = (double) f[i];
                }

                xfree(l2);
                l2 = NULL;
                
                stl_read_line(&l3, ptr);
                p = stl_str_cfie(ve, l3);

                if (3 != (r = sscanf(&l3[p + (int)strlen(ve)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", ve, n);

                pos[1] = 1;

                for (pos[0]= 0; pos[0] < 3; pos[0]++)
                        MD_ACCESS(D, strs, pos, model) = (double) f[pos[0]];

                xfree(l3);
                l3 = NULL;
                
                stl_read_line(&l4, ptr);
                p = stl_str_cfie(ve, l4);

                if (3 != (r = sscanf(&l4[p + (int)strlen(ve)], "%f %f %f", &f[0], &f[1], &f[2])))
                        error("reading %s values failed in %dth block.\n", ve, n);

                pos[1] = 2;

                for (pos[0] = 0; pos[0] < 3; pos[0]++)
                        MD_ACCESS(D, strs, pos, model) = (double) f[pos[0]];

                xfree(l4);
                l4 = NULL;
                
                stl_read_line(&l5, ptr);
                stl_str_cfie("endloop", l5);
                xfree(l5);
                l5 = NULL;
                
                stl_read_line(&l6, ptr);
                stl_str_cfie("endfacet", l6);
                xfree(l6);
                l6 = NULL;

        }
        stl_read_line(&l0, ptr);
        stl_str_cfie("endsolid", l0);
        xfree(l0);
        
        fclose(ptr);

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
static double* stl_read_binary(int D, long dims[D], const char* name) 
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
        
        md_set_dims(D, dims, 1);
        dims[0] = 3;
        dims[1] = 4;
        dims[2] = N;
        double* model = md_alloc(D, dims, D_SIZE);
        long strs[D];
        md_calc_strides(D, strs, dims, D_SIZE);

        long pos0[D];
        md_set_dims(D, pos0, 0);

        do {

                MD_ACCESS(D, strs, pos0, model) = 0;

        } while (md_next(D, dims, ~0UL, pos0));

        const size_t bs = 12 * FL_SIZE + sizeof(uint16_t);
        const size_t L = (size_t) N * bs;
        char* buf = xmalloc(L);
        
        if ((int) L != xread(fd, L, buf))
                error("stl file could not be opened\n");

        close(fd);

#pragma omp parallel for        
        for (int i = 0; i < N; i++) {

                long pos[D];
                md_set_dims(D, pos, 0);
                pos[2] = i;
                pos[1] = 3;

                for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                        float f;
                        memcpy(&f, &buf[(size_t) i * bs + (size_t) pos[0] * FL_SIZE], FL_SIZE);
                        MD_ACCESS(D, strs, pos, model) = f;
                }
                
                for (pos[1] = 0; pos[1] < 3; pos[1]++) {

                        for (pos[0] = 0; pos[0] < 3; pos[0]++) {

                                float f;
                                memcpy(&f, &buf[(size_t) i * bs + (size_t) (3 + pos[0] + 3 * pos[1]) * FL_SIZE], FL_SIZE);
                                MD_ACCESS(D, strs, pos, model) = f;
                        }
                }
        }
        xfree(buf);
        return model;
}

double* stl_read(int D, long dims[D], const char* name) 
{
        return stl_is_ascii(name) ? stl_read_ascii(D, dims, name) : stl_read_binary(D, dims, name);
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
double* stl_cfl2d(int D, long dims[D], complex float* cmodel)
{
        double* model = md_alloc(D, dims, D_SIZE);
        long pos[D], dstrs[D], cstrs[D];

        md_set_dims(D, pos, 0);
        md_calc_strides(D, dstrs, dims, D_SIZE);
        md_calc_strides(D, cstrs, dims, CFL_SIZE);

        do {
                MD_ACCESS(D, dstrs, pos, model) = (float) MD_ACCESS(D, cstrs, pos, cmodel);

        } while (md_next(D, dims, ~0UL, pos));

        return model;
}

// convert model in double md array to model in cfl md array
void stl_d2cfl(int D, long dims[D], double* model, complex float* cmodel)
{
        long dstrs[D], cstrs[D], pos[D];

        md_set_dims(D, pos, 0);
        md_calc_strides(D, dstrs, dims, D_SIZE);
        md_calc_strides(D, cstrs, dims, CFL_SIZE);

        do {

                MD_ACCESS(D, cstrs, pos, cmodel) = (float) MD_ACCESS(D, dstrs, pos, model) + 0.j;

        } while (md_next(D, dims, ~0UL, pos));
}

#ifndef _STLMISC_H
#define _STLMISC_H

#include "num/flpmath.h"

#define TOL 1E-14

enum stl_itype { STL_NONE, STL_TETRAHEDRON, STL_HEXAHEDRON };

// contains triangles of stl files and rotation/ shift rel. to origin
struct triangle {

	// vertices and normal vector
	double v0[3];
	double v1[3];
	double v2[3];
	double n[3];

	double e0[3];
	double e1[3];

	// center
	double ctr[3];
	// rotation axis and angle for rotation in x y plane
	double rot[3];
	double angle;

	// signed volume of tetrahendron triangle + origin
	double svol;
	// surface measure of triangle
	double sur;

	// intersection coordinates of triangle with 2d plane
	double poly[6];
};

struct triangle_stack {

	int N;
	void* tri;
};

extern struct triangle triangle_defaults;
extern struct triangle_stack triangle_stack_defaults;

extern void stl_unormal_vec3(double* n, const double* v0, const double* v1);
extern void stl_sub_vec3(double* o, const double* v0, const double* v1);
extern double stl_norm_vec(int N, const double* d);
extern void print_vec(int N, const double* d);

extern void stl_center_fov(int D, long dims[D], double* model, double fov_size);
extern void stl_stats(int D, long dims[D], double* model);
extern void stl_print(int D, long dims[D], double* model);
extern void stl_compute_normals(int D, long dims[D], double* model);
extern void stl_shift_model(int D, long dims[D], double* model, double shift[3]);
extern void stl_scale_model(int D, long dims[D], double* model, double scale[3]);

extern _Bool stl_fileextension(const char* name);
extern double* stl_read(int D, long dims[D], const char* name);
extern void stl_write_binary(int D, long dims[D], double* model, const char* name);
extern double* stl_cfl2d(int D, long dims[D], _Complex float* cmodel);
extern void stl_d2cfl(int D, long dims[D], double* model, _Complex float* cmodel);
extern void stl_relative_position(struct triangle* t);
extern struct triangle_stack* stl_preprocess_model(int D, long dims[D], double* model);

#endif

#ifndef _STLMISC_H
#define _STLMISC_H

#include "num/flpmath.h"

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
	struct triangle* tri;
};

extern struct triangle triangle_defaults;
extern struct triangle_stack triangle_stack_defaults;

extern void stl_center_fov(const long dims[3], double* model, double fov_size);
extern void stl_stats(const long dims[3], const double* model);
extern void stl_print(const long dims[3], const double* model);
extern void stl_compute_normals(const long dims[3], double* model);
extern void stl_shift_model(const long dims[3], double* model, const double shift[3]);
extern void stl_scale_model(const long dims[3], double* model, const double scale[3]);

extern _Bool stl_fileextension(const char* name);
extern double* stl_read(const char* name, long dims[3]);
extern void stl_write_binary(const char* name, const long dims[3], const double* model);
extern double* stl_cfl2d(const long dims[3], const _Complex float* cmodel);
extern void stl_d2cfl(const long dims[3], _Complex float* cmodel, const double* model);
extern void stl_relative_position(struct triangle* t);
extern struct triangle_stack* stl_preprocess_model(const long dims[3], const double* model);

#endif

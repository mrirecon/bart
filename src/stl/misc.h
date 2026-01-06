#ifndef _STLMISC_H
#define _STLMISC_H

#define TOL 1E-14

#define D_SIZE      sizeof(double)
#define FL_SIZE     sizeof(float)
#define CFL_SIZE	sizeof(_Complex float)

enum stl_itype { STL_NONE, STL_TETRAHEDRON, STL_HEXAHEDRON };

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


#endif

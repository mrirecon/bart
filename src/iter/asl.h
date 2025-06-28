#ifndef _ITER_ASL_H
#define _ITER_ASL_H

extern void get_asl_dims(int N, int asl_dim, long asl_dims[N], const long in_dims[N]);
extern const struct linop_s* linop_asl_create(int N, const long img_dims[N], int asl_dim);

extern void get_teasl_label_dims(int N, int teasl_dim, long teasl_label_dims[N], const long in_dims[N]);
extern void get_teasl_pwi_dims(int N, int teasl_dim, long teasl_pwi_dims[N], const long in_dims[N]);
extern const struct linop_s* linop_teasl_extract_label(int N, const long img_dims[N], int teasl_dim);
extern const struct linop_s* linop_teasl_extract_pwi(int N, const long img_dims[N], int teasl_dim);

#endif // _ITER_ASL_H

#ifndef __ASL_H
#define __ASL_H

extern void get_asl_dims(int N, int asl_dim, long asl_dims[N], const long in_dims[N]);
extern const struct linop_s* linop_asl_create(int N, const long img_dims[N], int asl_dim);

#endif // __ASL_H

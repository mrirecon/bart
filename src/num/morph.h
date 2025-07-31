
#include "num/conv.h"

extern _Complex float* md_structuring_element_cube(int N, long dims[N], int radius, unsigned long flags, const void* ref);
extern _Complex float* md_structuring_element_ball(int N, long dims[N], int radius, unsigned long flags, const void* ref);
extern _Complex float* md_structuring_element_cross(int N, long dims[N], int radius, unsigned long flags, const void* ref);

extern void md_erosion(int D, const long mask_dims[D], _Complex float* mask, const long dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);
extern void md_dilation(int D, const long mask_dims[D], _Complex float* mask, const long dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);
extern void md_opening(int D, const long mask_dims[D], _Complex float* mask, const long dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);
extern void md_closing(int D, const long mask_dims[D], _Complex float* mask, const long dims[D], _Complex float* out, const _Complex float* in, enum conv_type ctype);

extern _Complex float* md_label_simple_connection(int N, long dims[N], float radius, unsigned long flags);
extern long md_label(int N, const long dims[N], _Complex float* labels, const _Complex float* src, const long sdims[N], const _Complex float* structure);

void md_center_of_mass(int N_labels, int N, float com[N_labels][N], const long dims[N], const _Complex float* labels, const _Complex float* wgh);

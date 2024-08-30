
extern void md_positions(int N, int d, unsigned long flags, const long sdims[__VLA(N)], const long pdims[__VLA(N)], _Complex float* pos);

extern void md_interpolate2(int d, unsigned long flags, int ord, int N, const long dims[__VLA(N)], const long istrs[__VLA(N)], _Complex float* intp, const long cstrs[__VLA(N)], const _Complex float* coor, const long gdims[__VLA(N)], const long gstrs[__VLA(N)], const _Complex float* grid);
extern void md_interpolateH2(int d, unsigned long flags, int ord, int N, const long gdims[__VLA(N)], const long gstrs[__VLA(N)], _Complex float* grid, const long dims[__VLA(N)], const long istrs[__VLA(N)], const _Complex float* intp, const long cstrs[__VLA(N)], const _Complex float* coor);
extern void md_interpolate_adj_coor2(int d, unsigned long flags, int ord, int N, const long dims[__VLA(N)], const long cstrs[__VLA(N)], const _Complex float* coor, _Complex float* dcoor, const long istrs[__VLA(N)], const _Complex float* dintp, const long gdims[__VLA(N)], const long gstrs[__VLA(N)], const complex float* grid);
extern void md_interpolate_der_coor2(int d, unsigned long flags, int ord, int N, const long dims[__VLA(N)], const long istrs[__VLA(N)], _Complex float* dintp, const long cstrs[__VLA(N)], const _Complex float* coor, const _Complex float* dcoor, const long gdims[__VLA(N)], const long gstrs[__VLA(N)], const _Complex float* grid);

extern void md_interpolate(int d, unsigned long flags, int ord, int N, const long idims[__VLA(N)], _Complex float* intp, const long cdims[__VLA(N)], const _Complex float* coor, const long gdims[__VLA(N)], const _Complex float* grid);
extern void md_interpolateH(int d, unsigned long flags, int ord, int N, const long gdims[__VLA(N)], _Complex float* grid, const long idims[__VLA(N)], const _Complex float* intp, const long cdims[__VLA(N)], const _Complex float* coor);
extern void md_interpolate_adj_coor(int d, unsigned long flags, int ord, int N, const long cdims[__VLA(N)], const _Complex float* coor, _Complex float* dcoor, const long idims[__VLA(N)], const _Complex float* dintp, const long gdims[__VLA(N)], const _Complex float* grid);
extern void md_interpolate_der_coor(int d, unsigned long flags, int ord, int N, const long idims[__VLA(N)], _Complex float* dintp, const long cdims[__VLA(N)], const _Complex float* coor, const _Complex float* dcoor, const long gdims[__VLA(N)], const _Complex float* grid);

extern void md_resample(unsigned long flags, int ord, int N, const long odims[__VLA(N)], _Complex float* dst, const long idims[__VLA(N)], const _Complex float* src);

struct linop_s;
extern const struct linop_s* linop_interpolate_create(int d, unsigned long flags, int ord, int N, const long idims[__VLA(N)], const long cdims[__VLA(N)], const _Complex float* coor, const long gdims[__VLA(N)]);

struct nlop_s;
extern const struct nlop_s* nlop_interpolate_create(int d, unsigned long flags, int ord, _Bool shifted_grad, int N, const long idims[__VLA(N)], const long cdims[__VLA(N)], const long gdims[__VLA(N)]);


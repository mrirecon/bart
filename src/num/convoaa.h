
#include <complex.h>

extern void overlapandadd(int N, const long dims[N], const long blk[N], complex float* dst, complex float* src1, const long dim2[N], complex float* src2);
extern void overlapandsave(int N, const long dims[N], const long blk[N], complex float* dst, complex float* src1, const long dim2[N], complex float* src2);

extern struct conv_plan* overlapandsave_plan(int N, const long dims[N], const long blk[N], const long dim2[N], complex float* src2);
extern void overlapandsave_exec(struct conv_plan* plan, int N, const long dims[N], const long blk[N], complex float* dst, complex float* src1, const long dim2[N]);

extern void overlapandsave2(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], const complex float* src1, const long dims2[N], const complex float* src2);
extern void overlapandsave2H(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], const complex float* src1, const long dims2[N], const complex float* src2);
extern void overlapandsave2NE(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], complex float* src1, const long dims2[N], complex float* src2, const long mdims[N], complex float* msk);

struct vec_ops;
extern void overlapandsave2NEB(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], const complex float* src1, const long dims2[N], const complex float* src2, const long mdims[N], const complex float* msk);

extern void overlapandsave2HB(int N, unsigned int flags, const long blk[N], const long odims[N], complex float* dst, const long dims1[N], const complex float* src1, const long dims2[N], const complex float* src2, const long mdims[N], const complex float* msk);



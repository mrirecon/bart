
struct fft_cuda_plan_s;

extern struct fft_cuda_plan_s* fft_cuda_plan(int D, const long dimensions[D], unsigned long flags, const long ostrides[D], const long istrides[D], _Bool dir);
extern void fft_cuda_free_plan(struct fft_cuda_plan_s* cuplan);
extern void fft_cuda_exec(struct fft_cuda_plan_s* cuplan, complex float* dst, const complex float* src);


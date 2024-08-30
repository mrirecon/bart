extern void optical_flow(_Bool l1_reg, unsigned long reg_flags, float lambda, float maxnorm, _Bool l1_dc, int d, unsigned long flags, int N, const long dims[N], const _Complex float* img_static, const _Complex float* _img_moved, _Complex float* u);

extern void optical_flow_multiscale(_Bool l1_reg, unsigned long reg_flags, float lambda, float maxnorm, _Bool l1_dc, 
				    int levels, float sigma[levels], float factors[levels], int nwarps[levels],
				    int d, unsigned long flags, int N, const long _dims[N], const _Complex float* img_static, const _Complex float* img_moved, _Complex float* u);

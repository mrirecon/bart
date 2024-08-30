
extern void gaussian_pyramide(int levels, float factors[levels], float sigma[levels], int order,
			      int N, unsigned long flags, const long idims[N], const _Complex float* img,
			      long dims[levels][N], _Complex float* imgs[levels]);

extern void debug_gaussian_pyramide(int levels, float factors[levels], float sigma[levels],
				int N, unsigned long flags, const long idims[N]);

extern void upscale_displacement(int N, int d, unsigned long flags,
				 const long odims[N], _Complex float* out,
				 const long idims[N], const _Complex float* in);

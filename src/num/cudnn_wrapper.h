void cudnn_init(void);
void cudnn_deinit(void);

extern _Bool zconvcorr_fwd_cudnn(	int N,
					long odims[N], long ostrs[N], _Complex float* out,
					long idims[N], long istrs[N], const _Complex float* in,
					long kdims[N], long kstrs[N], const _Complex float* krn,
					unsigned long flags, const long dilation[N], const long strides[N], _Bool conv);

extern _Bool zconvcorr_bwd_in_cudnn(	int N,
					long odims[N], long ostrs[N], const _Complex float* out,
					long idims[N], long istrs[N], _Complex float* in,
					long kdims[N], long kstrs[N], const _Complex float* krn,
					unsigned long flags, const long dilation[N], const long strides[N], _Bool conv);

extern _Bool zconvcorr_bwd_krn_cudnn(	int N,
					long odims[N], long ostrs[N], const _Complex float* out,
					long idims[N], long istrs[N], const _Complex float* in,
					long kdims[N], long kstrs[N], _Complex float* krn,
					unsigned long flags, const long dilation[N], const long strides[N], _Bool conv);
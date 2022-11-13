
#ifndef __SOMEOPS_H
#define __SOMEOPS_H

#include <stdbool.h>

#include "misc/cppwrap.h"

extern struct linop_s* linop_cdiag_create(unsigned int N, const long dims[__VLA(N)], unsigned int flags, const _Complex float* diag);
extern struct linop_s* linop_rdiag_create(unsigned int N, const long dims[__VLA(N)], unsigned int flags, const _Complex float* diag);
extern void linop_gdiag_set_diag(const struct linop_s* lop, int N, const long ddims[__VLA(N)], const _Complex float* diag);
extern struct linop_s* linop_scale_create(unsigned int N, const long dims[N], const _Complex float scale);
extern struct linop_s* linop_zconj_create(unsigned int N, const long dims[N]);
extern struct linop_s* linop_zreal_create(unsigned int N, const long dims[N]);
extern struct linop_s* linop_flip_create(int N, const long dims[N], unsigned long flags);

extern struct linop_s* linop_identity_create(unsigned int N, const long dims[__VLA(N)]);
extern _Bool linop_is_identity(const struct linop_s* lop);

extern struct linop_s* linop_resize_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);	// deprecated
extern struct linop_s* linop_resize_center_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);
extern struct linop_s* linop_expand_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);
extern struct linop_s* linop_reshape_create(unsigned int A, const long out_dims[__VLA(A)], int B, const long in_dims[__VLA(B)]);
extern struct linop_s* linop_extract_create(unsigned int N, const long pos[N], const long out_dims[N], const long in_dims[N]);
extern struct linop_s* linop_permute_create(unsigned int N, const int order[__VLA(N)], const long idims[N]);
extern struct linop_s* linop_transpose_create(int N, int a, int b, const long dims[N]);


extern struct linop_s* linop_fft_create(int N, const long dims[__VLA(N)], unsigned int flags);
extern struct linop_s* linop_ifft_create(int N, const long dims[__VLA(N)], unsigned int flags);
extern struct linop_s* linop_fftc_create(int N, const long dims[__VLA(N)], unsigned int flags);
extern struct linop_s* linop_ifftc_create(int N, const long dims[__VLA(N)], unsigned int flags);

extern struct linop_s* linop_cdf97_create(int N, const long dims[__VLA(N)], unsigned int flag);

#ifndef __PADD_ENUMS
#define __PADD_ENUMS
enum PADDING { PAD_VALID, PAD_SAME, PAD_CYCLIC, PAD_SYMMETRIC, PAD_REFLECT, PAD_CAUSAL };
#endif
extern struct linop_s* linop_padding_create_onedim(int N, const long dims[N], enum PADDING pad_type, int pad_dim, long pad_for, long pad_after);
extern struct linop_s* linop_padding_create(int N, const long dims[N], enum PADDING pad_type, long pad_for[N], long pad_after[N]);

#ifndef __CONV_ENUMS
#define __CONV_ENUMS
enum conv_mode { CONV_SYMMETRIC, CONV_CAUSAL, CONV_ANTICAUSAL };
enum conv_type { CONV_CYCLIC, CONV_TRUNCATED, CONV_VALID, CONV_EXTENDED };
#endif

extern struct linop_s* linop_conv_create(int N, unsigned int flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)],
                const long idims1[__VLA(N)], const long idims2[__VLA(N)], const _Complex float* src2);

extern struct linop_s* linop_matrix_create(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)], const long matrix_dims[__VLA(N)], const _Complex float* matrix);
extern struct linop_s* linop_matrix_altcreate(unsigned int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)], const unsigned int T_dim, const unsigned int K_dim, const _Complex float* matrix);


extern struct linop_s* linop_matrix_chain(const struct linop_s* a, const struct linop_s* b);

#include "misc/cppwrap.h"
#endif // __SOMEOPS_H

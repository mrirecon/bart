
#ifndef _LINOPS_SOMEOPS_H
#define _LINOPS_SOMEOPS_H

#include <stdbool.h>

#include "misc/cppwrap.h"

extern struct linop_s* linop_cdiag_create(int N, const long dims[__VLA(N)], unsigned long flags, const _Complex float* diag);
extern struct linop_s* linop_rdiag_create(int N, const long dims[__VLA(N)], unsigned long flags, const _Complex float* diag);
extern void linop_gdiag_set_diag(const struct linop_s* lop, int N, const long ddims[__VLA(N)], const _Complex float* diag);
extern void linop_gdiag_set_diag_F(const struct linop_s* lop, int N, const long ddims[__VLA(N)], const _Complex float* diag);
extern void linop_gdiag_set_diag_ref(const struct linop_s* lop, int N, const long ddims[__VLA(N)], const _Complex float* diag);

extern struct linop_s* linop_scale_create(int N, const long dims[N], const _Complex float scale);
extern struct linop_s* linop_zconj_create(int N, const long dims[N]);
extern struct linop_s* linop_zreal_create(int N, const long dims[N]);
extern struct linop_s* linop_flip_create(int N, const long dims[N], unsigned long flags);

extern struct linop_s* linop_identity_create(int N, const long dims[__VLA(N)]);
extern _Bool linop_is_identity(const struct linop_s* lop);

extern struct linop_s* linop_copy_block_create(int N, const long pos[__VLA(N)], const long odims[__VLA(N)], const long idims[__VLA(N)]);
extern struct linop_s* linop_resize_create(int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);	// deprecated
extern struct linop_s* linop_resize_center_create(int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);
extern struct linop_s* linop_expand_create(int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);
extern struct linop_s* linop_reshape_create(int A, const long out_dims[__VLA(A)], int B, const long in_dims[__VLA(B)]);
extern struct linop_s* linop_reshape2_create(int N, unsigned long flags, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)]);
extern struct linop_s* linop_extract_create(int N, const long pos[N], const long out_dims[N], const long in_dims[N]);
extern struct linop_s* linop_permute_create(int N, const int order[__VLA(N)], const long idims[N]);
extern struct linop_s* linop_transpose_create(int N, int a, int b, const long dims[N]);

extern struct linop_s* linop_slice_create(int N, unsigned long flags, const long pos[__VLA(N)], const long dims[__VLA(N)]);
extern struct linop_s* linop_slice_one_create(int N, int idx, long pos, const long dims[__VLA(N)]);

extern struct linop_s* linop_add_strided_create(int N, const long dims[__VLA(N)], const long ostrs[__VLA(N)], const long istrs[__VLA(N)],
					        int OO, const long odims[__VLA(OO)], int II, const long idims[__VLA(II)]);
extern struct linop_s* linop_hankelization_create(int N, const long dims[__VLA(N)], int dim, int window_dim, int window_size);


extern struct linop_s* linop_fft_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern struct linop_s* linop_ifft_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern struct linop_s* linop_fftc_create(int N, const long dims[__VLA(N)], unsigned long flags);
extern struct linop_s* linop_ifftc_create(int N, const long dims[__VLA(N)], unsigned long flags);

extern struct linop_s* linop_fft_generic_create(int N, const long dims[N], unsigned long flags, unsigned long center_flags, unsigned long unitary_flags, unsigned long pre_flag, const _Complex float* pre_diag, unsigned long post_flag, const _Complex float* post_diag);
extern struct linop_s* linop_ifft_generic_create(int N, const long dims[N], unsigned long flags, unsigned long center_flags, unsigned long unitary_flags, unsigned long pre_flag, const _Complex float* pre_diag, unsigned long post_flag, const _Complex float* post_diag);

extern struct linop_s* linop_cdf97_create(int N, const long dims[__VLA(N)], unsigned long flag);

#ifndef _PADD_ENUMS
#define _PADD_ENUMS
enum PADDING { PAD_VALID, PAD_SAME, PAD_CYCLIC, PAD_SYMMETRIC, PAD_REFLECT, PAD_CAUSAL };
#endif
extern struct linop_s* linop_padding_create_onedim(int N, const long dims[N], enum PADDING pad_type, int pad_dim, long pad_for, long pad_after);
extern struct linop_s* linop_padding_create(int N, const long dims[N], enum PADDING pad_type, long pad_for[N], long pad_after[N]);

extern struct linop_s* linop_shift_create(int N, const long dims[__VLA(N)], int shift_dim, long shift, enum PADDING pad_type);

#ifndef _CONV_ENUMS
#define _CONV_ENUMS
enum conv_mode { CONV_SYMMETRIC, CONV_CAUSAL, CONV_ANTICAUSAL };
enum conv_type { CONV_CYCLIC, CONV_TRUNCATED, CONV_VALID, CONV_EXTENDED };
#endif

extern struct linop_s* linop_conv_create(int N, unsigned long flags, enum conv_type ctype, enum conv_mode cmode, const long odims[__VLA(N)],
                const long idims1[__VLA(N)], const long idims2[__VLA(N)], const _Complex float* src2);

extern struct linop_s* linop_conv_gaussian_create(int N, enum conv_type ctype, const long dims[__VLA(N)], const float sigma[__VLA(N)]);

extern struct linop_s* linop_matrix_create(int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)], const long matrix_dims[__VLA(N)], const _Complex float* matrix);
extern struct linop_s* linop_matrix_altcreate(int N, const long out_dims[__VLA(N)], const long in_dims[__VLA(N)], int T_dim, int K_dim, const _Complex float* matrix);


extern struct linop_s* linop_matrix_chain(const struct linop_s* a, const struct linop_s* b);

extern struct linop_s* linop_hadamard_create(int N, const long in_dims[__VLA(N)], int hadamard_dim);

#include "misc/cppwrap.h"
#endif // _LINOPS_SOMEOPS_H

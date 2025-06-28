
#ifndef _PADD_ENUMS
#define _PADD_ENUMS
enum PADDING { PAD_VALID, PAD_SAME, PAD_CYCLIC, PAD_SYMMETRIC, PAD_REFLECT, PAD_CAUSAL };
#endif

extern struct nlop_s* nlop_convcorr_geom_create(int N, unsigned long flags, const long odims[N], const long idims[N], const long kdims[N],
						enum PADDING conv_pad, _Bool conv, const long strides[N], const long dilations[N], char transpc);


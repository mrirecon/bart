
#include "misc/cppwrap.h"

struct pytorch_wrapper_s;

extern struct pytorch_wrapper_s* pytorch_wrapper_create(const char* path, int II, const int DI[__VLA(II)], const long* idims[__VLA(II)], int device);

extern void pytorch_wrapper_free(const struct pytorch_wrapper_s* data);

extern int pytorch_wrapper_number_outputs(const struct pytorch_wrapper_s* data);
extern int pytorch_wrapper_rank_output(const struct pytorch_wrapper_s* data, int o);
extern void pytorch_wrapper_dims_output(const struct pytorch_wrapper_s* data, int o, int N, long dims[__VLA(N)]);

extern void pytorch_wrapper_apply_unchecked(struct pytorch_wrapper_s* data, int N, _Complex float* args[__VLA(N)], int device);
extern void pytorch_wrapper_adjoint_unchecked(struct pytorch_wrapper_s* data, int o, int i, _Complex float* dst, const _Complex float* src);
extern void pytorch_wrapper_derivative_unchecked(struct pytorch_wrapper_s* data, int o, int i, _Complex float* dst, const _Complex float* src);

#include "misc/cppwrap.h"

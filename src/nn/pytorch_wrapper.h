#include "misc/cppwrap.h"

struct nlop_s;
extern const struct nlop_s* nlop_pytorch_create(const char* path, int II, const int DI[__VLA(II)], const long* idims[__VLA(II)], bool init_gpu);

#include "misc/cppwrap.h"
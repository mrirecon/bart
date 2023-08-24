#include "misc/cppwrap.h"

struct nlop_s;
extern const struct nlop_s* nlop_external_graph_create(const char* path, int OO, const int DO[__VLA2(OO)], const long* odims[__VLA2(OO)], int II, const int DI[__VLA(II)], const long* idims[__VLA(II)], _Bool init_gpu, const char* tf_key);

#include "misc/cppwrap.h"

#include <stdbool.h>

struct TF_Tensor;

struct nlop_s;
extern const struct nlop_s* nlop_tf_create(int nr_outputs, int nr_inputs, const char* path, bool session);


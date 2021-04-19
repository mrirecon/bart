
#include <stdbool.h>

struct TF_Tensor;

struct nlop_s;
extern const struct nlop_s* nlop_tf_create(int nr_outputs, int nr_inputs, const char* path, bool session);
extern struct TF_Tensor* const* get_input_tensor(const struct nlop_s*);


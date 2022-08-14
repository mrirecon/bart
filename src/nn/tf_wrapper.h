
#include <stdbool.h>

struct TF_Tensor;

struct nlop_s;

struct tf_shared_graph_s;

extern const struct tf_shared_graph_s* tf_shared_graph_create(const char* path, const char* signature_key);
extern void tf_shared_graph_free(const struct tf_shared_graph_s* x);
extern const char* tf_shared_graph_get_init_path(const struct tf_shared_graph_s* x);


extern const struct nlop_s* nlop_tf_shared_create(const struct tf_shared_graph_s* graph);
extern const struct nlop_s* nlop_tf_create(const char* path);


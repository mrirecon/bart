#include "networks/cnn.h"

struct tf_shared_graph_s;

struct network_tensorflow_s {

	INTERFACE(network_t);

	const char* model_path;

	const struct tf_shared_graph_s* tf_graph;
};

extern struct network_tensorflow_s network_tensorflow_default;

extern nn_t network_tensorflow_create(const struct network_s* config, unsigned int NO, const long odims[NO], unsigned int NI, const long idims[NI], enum NETWORK_STATUS status);

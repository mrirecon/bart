
#include "linops/someops.h"
#include "nn/activation.h"

struct nn_cunet_conf_s {

	int levels;

	enum PADDING padding;
	enum ACTIVATION activation;

	bool conditional;
	long num_filters;
	long cunits;

	long ksizes[3];
	long dilations[3];
	long strides[3];
};

extern struct nn_cunet_conf_s cunet_defaults;

extern const struct nn_s* cunet_create(struct nn_cunet_conf_s* conf, int N, const long dims[N]);

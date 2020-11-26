#ifndef NN_WEIGHTS_H
#define NN_WEIGHTS_H

#include "nn/nn.h"

struct iovec_s;
struct nlop_s;
struct nn_s;

struct nn_weights_s {

	int N;
	const struct iovec_s** iovs;
	_Complex float** tensors;
};

const struct nn_weights_s* create_multi_md_array(int N, int D[N], const long* dimensions[N], const _Complex float* x[N], size_t sizes[N]);
void free_multi_md_array(const struct nn_weights_s* array);

typedef struct nn_weights_s* nn_weights_t;

nn_weights_t nn_weights_create(int N, const struct iovec_s* iovs[N]);
nn_weights_t nn_weights_create_from_nn(nn_t x);
nn_weights_t load_nn_weights(const char *name);

extern void nn_weights_copy(nn_weights_t dst, nn_weights_t src);

void dump_nn_weights(const char *name, nn_weights_t weights);
void move_gpu_nn_weights(nn_weights_t weights);
_Bool nn_weights_on_gpu(nn_weights_t weights);
void nn_weights_free(nn_weights_t weights);

void nn_init(nn_t op, nn_weights_t weights);

const struct nn_s* nn_get_wo_weights(nn_t op, nn_weights_t weights, _Bool copy);
const struct nn_s* nn_get_wo_weights_F(nn_t op, nn_weights_t weights, _Bool copy);

const struct nlop_s* nn_get_nlop_wo_weights(nn_t op, nn_weights_t weights, _Bool copy);
const struct nlop_s* nn_get_nlop_wo_weights_F(nn_t op, nn_weights_t weights, _Bool copy);

#endif

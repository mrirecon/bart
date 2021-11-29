#ifndef NN_DATA_LIST_H
#define NN_DATA_LIST_H

#include "iter/batch_gen.h"

#include "nn/nn.h"

struct named_data_list_s;

extern struct named_data_list_s* named_data_list_create(void);
extern void named_data_list_free(struct named_data_list_s* data_list);
extern void named_data_list_append(struct named_data_list_s* data_list, int N, const long dims[N], _Complex float* data, const char* name);

extern const struct iovec_s* named_data_list_get_iovec(struct named_data_list_s* data_list, const char* name);
extern const struct nlop_s* nn_batchgen_create(nn_t network, struct named_data_list_s* train_data, enum BATCH_GEN_TYPE type, unsigned int seed);
extern nn_t nn_valid_create(nn_t network, struct named_data_list_s* valid_data);

#endif
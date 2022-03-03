#ifndef BATCHGEN_H
#define BATCHGEN_H

#include "misc/cppwrap.h"

enum BATCH_GEN_TYPE {
	BATCH_GEN_SAME,			// batches: 1, 2, 3 | 4, 5, 6 | 7, 8, 9
	BATCH_GEN_SHUFFLE_BATCHES,	// batches: 4, 5, 6 | 7, 8, 9 | 1, 2, 3
	BATCH_GEN_SHUFFLE_DATA,		// batches: 4, 8, 7 | 1, 3, 6 | 2, 5, 9
	BATCH_GEN_RANDOM_DATA		// batches: 7, 3, 7 | 4, 9, 9 | 1, 8, 3
	};
struct iter6_conf_s;

struct bat_gen_conf_s {

	long Nc;

	enum BATCH_GEN_TYPE type;
	unsigned int seed;

	unsigned long bat_flags;
};

extern struct bat_gen_conf_s bat_gen_conf_default;

extern const struct nlop_s* batch_generator_create2(struct bat_gen_conf_s* config, int D, int N, const long bat_dims[D][N], const long tot_dims[D][N], const long tot_strs[D][N], const _Complex float* data[D]);
extern const struct nlop_s* batch_generator_create(struct bat_gen_conf_s* config, int D, int N, const long bat_dims[D][N], const long tot_dims[D][N], const _Complex float* data[D]);

extern const struct nlop_s* batch_gen_create_from_iter(struct iter6_conf_s* iter_conf, int D, const int Ns[__VLA(D)], const long* bat_dims[__VLA(D)], const long* tot_dims[__VLA(D)], const _Complex float* data[__VLA(D)], long Nc);
extern const struct nlop_s* batch_gen_create(int D, const int Ns[__VLA(D)], const long* bat_dims[__VLA(D)], const long* tot_dims[__VLA(D)], const _Complex float* data[__VLA(D)], long Nc, enum BATCH_GEN_TYPE type, unsigned int seed);

#endif
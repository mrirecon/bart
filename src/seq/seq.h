
#ifndef _SEQ_H
#define _SEQ_H

#include "misc/dllspec.h"
#include "misc/cppwrap.h"

#include "seq/event.h"

enum block {

	BLOCK_UNDEFINED,
	BLOCK_PRE,
	BLOCK_KERNEL_CHECK,
	BLOCK_KERNEL_PREPARE,
	BLOCK_KERNEL_NOISE,
	BLOCK_KERNEL_DUMMY ,
	BLOCK_KERNEL_IMAGE,
	BLOCK_POST
};


enum context {

	CONTEXT_NORMAL,
	CONTEXT_BINARY,
	CONTEXT_UPDATE
};




struct seq_state {

	enum block mode;
	long chrono_slice;
	enum context context;
	int seq_ut; //perform ut
	long pos[DIMS];
	double start_block;
};

struct seq_config;

struct bart_seq {

	int version_bart;
	int version_seq;

	struct seq_config* conf;
	struct seq_state* state;
	int N;
	struct seq_event* event;
	int P;
	struct rf_shape* rf_shape;
};

BARTLIB_API struct bart_seq* BARTLIB_CALL bart_seq_alloc(void);
BARTLIB_API void BARTLIB_CALL bart_seq_defaults(struct bart_seq* seq);
BARTLIB_API void BARTLIB_CALL bart_seq_free(struct bart_seq* seq);


BARTLIB_API extern int BARTLIB_CALL seq_sample_rf_shapes(int N, struct rf_shape pulse[__VLA(N)], const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_compute_gradients(int M, double gradients[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);

BARTLIB_API extern double BARTLIB_CALL seq_nco_freq(const struct seq_event* ev);
BARTLIB_API extern double BARTLIB_CALL seq_nco_phase(int set, const struct seq_event* ev);
BARTLIB_API extern double BARTLIB_CALL seq_pulse_scaling(const struct rf_shape* pulse);
BARTLIB_API extern double BARTLIB_CALL seq_pulse_norm_sum(const struct rf_shape* pulse);
BARTLIB_API extern void BARTLIB_CALL seq_cfl_to_sample(const struct rf_shape* pulse, int idx, float* mag, float* pha);

BARTLIB_API extern double BARTLIB_CALL seq_block_end(int N, const struct seq_event ev[__VLA(N)], enum block mode, double tr, double raster);
BARTLIB_API extern double BARTLIB_CALL seq_block_end_flat(int N, const struct seq_event ev[__VLA(N)], double raster);
BARTLIB_API extern double BARTLIB_CALL seq_block_rdt(int N, const struct seq_event ev[__VLA(N)], double raster);

BARTLIB_API extern int BARTLIB_CALL seq_block(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);
BARTLIB_API extern int BARTLIB_CALL seq_continue(struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_H



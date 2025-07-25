
#ifndef _SEQ_H
#define _SEQ_H

#include "misc/cppwrap.h"

#include "seq/event.h"
#include "seq/config.h"

extern int seq_sample_rf_shapes(int N, struct rf_shape pulse[__VLA(N)], const struct seq_config* seq);
extern void seq_compute_gradients(int M, double gradients[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);

extern long seq_block_end(int N, const struct seq_event ev[__VLA(N)], enum block mode, long tr);
extern long seq_block_end_flat(int N, const struct seq_event ev[__VLA(N)]);
extern long seq_block_rdt(int N, const struct seq_event ev[__VLA(N)]);

extern int seq_block(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);
extern int seq_continue(struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_H



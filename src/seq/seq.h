
#ifndef __SEQ_H
#define __SEQ_H

#include "misc/cppwrap.h"

#include "seq/event.h"
#include "seq/config.h"

int seq_sample_rf_shapes(int N, struct rf_shape pulse[__VLA(N)], const struct seq_config* seq);
void seq_compute_gradients(int M, double gradients[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);

long seq_block_end(int N, const struct seq_event ev[__VLA(N)], enum block mode, long tr);
long seq_block_end_flat(int N, const struct seq_event ev[__VLA(N)]);
long seq_block_rdt(int N, const struct seq_event ev[__VLA(N)]);

int seq_block(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);
int seq_continue(struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // __SEQ_H




#ifndef _SEQ_H
#define _SEQ_H

#include "misc/dllspec.h"
#include "misc/cppwrap.h"

#include "seq/event.h"
#include "seq/config.h"

BARTLIB_API extern int BARTLIB_CALL seq_sample_rf_shapes(int N, struct rf_shape pulse[__VLA(N)], const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_compute_gradients(int M, double gradients[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);


BARTLIB_API extern double BARTLIB_CALL idea_nco_freq(const struct seq_event* ev);
BARTLIB_API extern double BARTLIB_CALL idea_nco_phase(int set, const struct seq_event* ev);
BARTLIB_API extern double BARTLIB_CALL idea_pulse_scaling(const struct rf_shape* pulse);
BARTLIB_API extern double BARTLIB_CALL idea_pulse_norm_sum(const struct rf_shape* pulse);
BARTLIB_API extern void BARTLIB_CALL idea_cfl_to_sample(const struct rf_shape* pulse, int idx, float* mag, float* pha);

BARTLIB_API extern double BARTLIB_CALL seq_block_end(int N, const struct seq_event ev[__VLA(N)], enum block mode, double tr, double raster);
BARTLIB_API extern double BARTLIB_CALL seq_block_end_flat(int N, const struct seq_event ev[__VLA(N)], double raster);
BARTLIB_API extern double BARTLIB_CALL seq_block_rdt(int N, const struct seq_event ev[__VLA(N)], double raster);

BARTLIB_API extern int BARTLIB_CALL seq_block(int N, struct seq_event ev[__VLA(N)], struct seq_state* seq_state, const struct seq_config* seq);
BARTLIB_API extern int BARTLIB_CALL seq_continue(struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_H



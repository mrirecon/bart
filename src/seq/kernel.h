#ifndef _SEQ_KERNEL_H
#define _SEQ_KERNEL_H

#include "misc/cppwrap.h"

#include "seq/event.h"
#include "seq/config.h"

void linearize_events(int N, struct seq_event ev[__VLA(N)], double* start_block, enum block mode, double tr, double raster);

extern void compute_moment0(int M, float moments[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);
extern void compute_adc_samples(int D, const long adc_dims[__VLA(D)], _Complex float* adc, int N, const struct seq_event ev[__VLA(N)]);
extern void gradients_support(int M, double gradients[__VLA(M)][6], int N, const struct seq_event ev[__VLA(N)]);

#include "misc/cppwrap.h"

#endif // _SEQ_KERNEL_H

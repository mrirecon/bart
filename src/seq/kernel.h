#ifndef __SEQ_KERNEL_H
#define __SEQ_KERNEL_H

#include <complex.h>

#include "misc/cppwrap.h"

#include "seq/event.h"

void seq_compute_moment0(int M, float moments[__VLA(M)][3], double dt, int N, const struct seq_event ev[__VLA(N)]);
void seq_compute_adc_samples(int D, const long adc_dims[__VLA(D)], _Complex float* adc, int N, const struct seq_event ev[__VLA(N)]);
void seq_gradients_support(int M, double gradients[__VLA(M)][6], int N, const struct seq_event ev[__VLA(N)]);

#include "misc/cppwrap.h"

#endif // __SEQ_KERNEL_H

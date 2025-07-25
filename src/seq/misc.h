
#ifndef _SEQ_MISC_H
#define _SEQ_MISC_H

struct seq_config;

extern double slice_amplitude(const struct seq_config* seq);
extern double ro_amplitude(const struct seq_config* seq);
extern long round_up_GRT(double val_usec);

struct grad_trapezoid;
struct grad_limits;
extern int gradient_prepare_with_timing(struct grad_trapezoid* grad, double moment, struct grad_limits sys);

#endif // _SEQ_MISC_H


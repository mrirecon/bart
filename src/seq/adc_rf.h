
#ifndef _SEQ_ADC_RF_H
#define _SEQ_ADC_RF_H

#include "misc/cppwrap.h"

#include "seq/config.h"
#include "seq/event.h"

extern double flash_ex_calls(const struct seq_config* seq);
extern double idea_phase_nco(int set, const struct seq_event* ev);
extern double idea_pulse_scaling(const struct rf_shape* pulse);
extern double idea_pulse_norm_sum(const struct rf_shape* pulse);
extern void idea_cfl_to_sample(const struct rf_shape* pulse, int idx, float* mag, float* pha);

double rf_spoiling(int D, const long pos[__VLA(D)], const struct seq_config* seq);

int prep_rf_inv(struct seq_event* rf_ev, double start, const struct seq_config* seq);

int prep_rf_ex(struct seq_event* rf_ev, double start, double rf_spoil_phase,
		const struct seq_state* seq_state, const struct seq_config* seq);


double adc_time_to_echo(const struct seq_config* seq);
double adc_duration(const struct seq_config* seq);

int prep_adc(struct seq_event* adc_ev, double start, double rf_spoil_phase,
		const struct seq_state* seq_state, const struct seq_config* seq);

#include "misc/cppwrap.h"

#endif // _SEQ_ADC_RF_H


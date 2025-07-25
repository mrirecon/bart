#include <math.h>

#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/flash.h"
#include "seq/seq.h"

#include "seq/adc_rf.c"

#include "utest.h"

#define FLASH_EVENTS 14


static bool test_flash_events(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = BLOCK_KERNEL_NOISE;
	E = flash(E, ev, &seq_state, &seq);

	if ((FLASH_EVENTS - 1) != E) // no rf
		return false;

	seq_state.mode = BLOCK_KERNEL_DUMMY;
	E = flash(E, ev, &seq_state, &seq);
	if ((FLASH_EVENTS - 1) != E) //no adc
		return false;

	seq_state.mode = BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_events);


static bool test_flash_te(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	long te[seq.loop_dims[TE_DIM]];
	events_get_te(seq.loop_dims[TE_DIM], te, E, ev);
	if (0 != (seq.phys.te - te[0]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_te);


static bool test_flash_mom1(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	seq_state.mode = BLOCK_KERNEL_IMAGE;
	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	int e_adc = events_idx(0, SEQ_EVENT_ADC, E, ev);
	int e_rf = events_idx(0, SEQ_EVENT_PULSE, E, ev);

	double mom[3];
	moment_sum(mom, ev[e_adc].mid, E, ev);
	if (UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_rf].mid + 1e-12, E, ev);
	if (UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (UT_TOL < fabs(mom[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom1);



static bool test_flash_mom2(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = BLOCK_KERNEL_IMAGE;
	struct seq_config seq = seq_config_defaults;

	int E = 200;
	struct seq_event ev[E];

	E = flash(E, ev, &seq_state, &seq);

	if (FLASH_EVENTS != E)
		return false;

	const int samples = seq.phys.tr;
	float m0[samples][3];
	seq_compute_moment0(samples, m0, 1., E, ev);
	long adc_mid = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)].mid;

	if (UT_TOL < fabs( m0[adc_mid][0] + m0[adc_mid - 1][0]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom2);


static bool test_flash_phase(void)
{
	struct seq_state seq_state = { 0 };
	seq_state.mode = BLOCK_KERNEL_IMAGE;
	struct seq_config seq = seq_config_defaults;
	seq.geom.shift[0][0] = 10.;
	seq.geom.shift[0][1] = 20.;
	seq.geom.shift[0][2] = 30.;
	seq.enc.pe_mode = PEMODE_RATION_APPROX_GA;

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < 1000; i++) {

		seq_state.pos[TIME_DIM] = i;
		E = flash(E, ev, &seq_state, &seq);

		struct seq_event ev_rf = ev[events_idx(0, SEQ_EVENT_PULSE, E, ev)];
		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];
		
		double phase_rf = phase_clamp(ev_rf.pulse.phase);
		double phase_echo = phase_clamp(ev_adc.adc.phase - adc_nco_correction(ev_adc.adc.freq, seq.phys, seq.sys));
		
		if (UT_TOL < fabs(phase_echo - phase_rf))
			return false;
	}
		
	return true;
}

UT_REGISTER_TEST(test_flash_phase);


static bool test_raga_spokes(void)
{
	struct seq_state seq_state = { 0 };;
	struct seq_config seq = seq_config_defaults;

	const int expected_spokes = 8;
	seq.enc.pe_mode = PEMODE_RATION_APPROX_GA;
	seq.enc.order = SEQ_ORDER_AVG_OUTER;
	set_loop_dims_and_sms(&seq, 1, 1, 5, expected_spokes, 1, 1, 1, 1, 0, 1);

	const int max_E = 200;
	struct seq_event ev[max_E];

	int ctr = 0;
	do {

		int E = seq_block(max_E, ev,  &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (BLOCK_KERNEL_IMAGE == seq_state.mode)
			ctr++;

	} while (seq_continue(&seq_state, &seq));

	if (ctr != expected_spokes)
		return false;
	return true;
}

UT_REGISTER_TEST(test_raga_spokes);

static bool test_raga_spokes_full(void)
{
	struct seq_state seq_state = { 0 };;
	struct seq_config seq = seq_config_defaults;
	seq.enc.order = SEQ_ORDER_AVG_OUTER;

	const int spk = 377;
	seq.enc.pe_mode = PEMODE_RATION_APPROX_GA;
	set_loop_dims_and_sms(&seq, 1, 1, spk, spk, 1, 1, 1, 1, 0, 1);

	const int max_E = 200;
	struct seq_event ev[max_E];

	int ctr = 0;
	do {

		int E = seq_block(max_E, ev,  &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (BLOCK_KERNEL_IMAGE == seq_state.mode)
			ctr++;

	} while (seq_continue(&seq_state, &seq));

	if (ctr != spk)
		return false;
	return true;
}

UT_REGISTER_TEST(test_raga_spokes_full);


static bool test_block_minv(void)
{
	const enum block blocks[16] = { BLOCK_KERNEL_NOISE, BLOCK_PRE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE
	};

	struct seq_state seq_state = { 0 };;
	struct seq_config seq = seq_config_defaults;
	seq.enc.order = SEQ_ORDER_AVG_OUTER;
	seq.magn.ti = 100;
	seq.magn.mag_prep = PREP_IR_NON;

	int i = 0;

	const int max_E = 200;
	struct seq_event ev[max_E];

	set_loop_dims_and_sms(&seq, 1, 2, 3, 3, 1, 2, 1, 1, 0, 1);
	int pre_blocks = 0;

	do {

		int E = seq_block(max_E, ev, &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq_state.mode)
			return false;

		if ((BLOCK_KERNEL_IMAGE == seq_state.mode) && (FLASH_EVENTS != E))
			return false;

		if (BLOCK_PRE == seq_state.mode)
			pre_blocks++;

		if ((BLOCK_PRE == seq_state.mode) && (1. * seq.magn.ti != (ev[E - 1].end - ev[E - 1].start)))
			return false;

		i++;

	} while (seq_continue(&seq_state, &seq));

	if (2 != pre_blocks)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_minv);


static bool test_block_minv_multislice(void)
{
	const enum block blocks[21] = {
		BLOCK_KERNEL_NOISE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
		BLOCK_PRE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE, BLOCK_KERNEL_IMAGE,
	};

	struct seq_state seq_state = { 0 };;
	struct seq_config seq = seq_config_defaults;
	seq.enc.order = SEQ_ORDER_SEQ_MS;
	seq.magn.ti = 100;
	seq.magn.mag_prep = PREP_IR_NON;
	// seq.magn.inv_delay_time = 100;

	int i = 0;
	int inversions = 0;

	const int max_E = 200;
	struct seq_event ev[max_E];

	set_loop_dims_and_sms(&seq, 1, 2, 3, 3, 1, 2, 1, 1, 0, 1);

	do {

		int E = seq_block(max_E, ev, &seq_state, &seq);

		if (0 > E)
			return false;

		if (0 == E)
			continue;

		if (blocks[i] != seq_state.mode)
			return false;

		if ((BLOCK_KERNEL_IMAGE == seq_state.mode) && (FLASH_EVENTS != E))
			return false;

		if (BLOCK_PRE == seq_state.mode)
			inversions++;

		// correct ti in mag_prep block
		if ((BLOCK_PRE == seq_state.mode) && (2 == E) && (1. * seq.magn.ti != seq_block_end(E, ev, seq_state.mode, seq.phys.tr)))
			return false;

		i++;

	} while (seq_continue(&seq_state, &seq));

	if (4 != inversions)
		return false;

	return true;
}

UT_REGISTER_TEST(test_block_minv_multislice);

#include <math.h>

#include "num/multind.h"

#include "seq/config.h"
#include "seq/event.h"
#include "seq/kernel.h"
#include "seq/flash.h"
#include "seq/seq.h"
#include "seq/helpers.h"
#include "seq/opts.h"

#include "utest.h"

#define FLASH_EVENTS 14


static bool test_command(void)
{
	struct seq_config seq = seq_config_defaults;
	seq.geom.baseres = 250;

	if (!read_config_from_str(&seq, "bart seq --BR 200 --FOV 305\0"))
		return false;

	if (200 != seq.geom.baseres)
		return false;

	if (305 != seq.geom.fov)
		return false;

	return true;
}

UT_REGISTER_TEST(test_command);


static bool test_command2(void)
{
	struct seq_config seq = seq_config_defaults;
	seq.geom.baseres = 250;
	
	const char* filename = "/path/not/existing/file.config";

	if (!seq_read_config_from_file(&seq, filename))
		return true;

	return false;
}

UT_REGISTER_TEST(test_command2);


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

	double te[seq.loop_dims[TE_DIM]];
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
	if (1E-5 * UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_rf].mid + 1e-12, E, ev);
	if (1E-5 * UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (1E-5 * UT_TOL < fabs(mom[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom1);


static bool test_flash_mom1b(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.phys.te = 2E-3;
	seq.phys.dwell = 4.3E-6;

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
	if (1E-5 * UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_rf].mid + 1e-12, E, ev);
	if (1E-5 * UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (1E-5 * UT_TOL < fabs(mom[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom1b);


static bool test_flash_mom1c(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;
	seq.phys.te = 2E-3;
	seq.phys.dwell = 4.1E-6;

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
	if (1E-5 * UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_rf].mid + 1e-12, E, ev);
	if (1E-5 * UT_TOL < (fabs(mom[0]) + fabs(mom[1]) + fabs(mom[2])))
		return false;

	moment_sum(mom, ev[e_adc].start, E, ev);
	if (1E-5 * UT_TOL < fabs(mom[2]))
		return false;

	return true;
}

UT_REGISTER_TEST(test_flash_mom1c);


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

	const int samples = lround(1.E6 * seq.phys.tr);
	float m0[samples][3];
	compute_moment0(samples, m0, 1.E-6, E, ev);
	long adc_mid = 1E6 * ev[events_idx(0, SEQ_EVENT_ADC, E, ev)].mid;

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
	seq.geom.shift[0][0] = 10.E-3;
	seq.geom.shift[0][1] = 20.E-3;
	seq.geom.shift[0][2] = 30.E-3;

	int E = 200;
	struct seq_event ev[E];

	for (int i = 0; i < 1000; i++) {

		seq_state.pos[TIME_DIM] = i;
		E = flash(E, ev, &seq_state, &seq);

		struct seq_event ev_rf = ev[events_idx(0, SEQ_EVENT_PULSE, E, ev)];
		struct seq_event ev_adc = ev[events_idx(0, SEQ_EVENT_ADC, E, ev)];
		
		if (UT_TOL < fabs(ev_adc.adc.phase - ev_rf.pulse.phase))
			return false;
	}
		
	return true;
}

UT_REGISTER_TEST(test_flash_phase);


static bool test_raga_spokes(void)
{
	struct seq_state seq_state = { 0 };
	struct seq_config seq = seq_config_defaults;

	const int expected_spokes = 8;

	seq.loop_dims[PHS1_DIM] = 5;
	seq.loop_dims[TIME_DIM] = expected_spokes;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

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

	const int spk = 377;
	seq.loop_dims[PHS1_DIM] = spk;
	seq.loop_dims[TIME_DIM] = spk;

	seq_ui_interface_loop_dims(0, &seq, DIMS, seq.loop_dims);

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


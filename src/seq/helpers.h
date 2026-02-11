
#ifndef _SEQ_HELPERS_H
#define _SEQ_HELPERS_H

// intended to additionally include in sequence
// (besides event.h seq.h custom_ui.h)
// we do not allow _Complex and bool here

// DO NOT CHANGE THIS HEADER !

#include "misc/dllspec.h"
#include "misc/cppwrap.h"

#include "misc/mri.h"

#define SEQ_MAX_PARAMS_LONG 64
#define SEQ_MAX_PARAMS_DOUBLE 16

#define SEQ_MAX_NO_ECHOES 64
#define SEQ_MAX_SLICES 64
#define SEQ_MAX_GRAD_POINTS 8192
#define SEQ_ACOUSTIC_RESONANCE_ENTRIES 5


enum seq_order {

	SEQ_ORDER_AVG_OUTER,
	SEQ_ORDER_AVG_INNER,
	SEQ_ORDER_SEQ_MS,
};

enum mag_prep {

	SEQ_PREP_OFF,
	SEQ_PREP_IR_SELECTIVE,
	SEQ_PREP_IR_NONSELECTIVE,
	SEQ_PREP_SR_SELECTIVE,
	SEQ_PREP_SR_NONSELECTIVE,
	SEQ_PREP_SR_ADIABATIC
};


enum trigger_type {

	SEQ_TRIGGER_OFF,
	SEQ_TRIGGER_ECG,
	SEQ_TRIGGER_PULSE,
	SEQ_TRIGGER_RESP,
	SEQ_TRIGGER_EXT,
};

struct seq_standard_conf {

	double tr;
	double te[SEQ_MAX_NO_ECHOES];

	double dwell;
	double flip_angle;

	double fov;
	int baseres;
	double slice_thickness;
	double slice_os;

	int is3D;

	double gamma;
	double b0;
	double grad_max_ampl;
	double grad_min_rise_time;
	double coil_control_lead;
	double min_duration_ro_rf;
	double raster_grad;
	double raster_rf;
	double raster_dwell;
	double acoustic_res_freq[SEQ_ACOUSTIC_RESONANCE_ENTRIES];
	double acoustic_res_bw[SEQ_ACOUSTIC_RESONANCE_ENTRIES];

	enum mag_prep mag_prep;
	double ti;

	enum trigger_type trigger_type;
	double trigger_delay_time;
	int trigger_pulses;
	int trigger_out;

	enum seq_order enc_order;
};

enum mode_flags {

	SEQ_MODE_3D		= (1u << 0),
	SEQ_MODE_BSSFP		= (1u << 1),
	SEQ_MODE_ASL		= (1u << 2),
	SEQ_MODE_INTERACTIVE	= (1u << 3),
};

struct seq_interface_conf {

	unsigned long mode;

	double tr;
	long radial_views;
	long slices;
	long echoes;
	double slice_thickness;

	enum trigger_type trigger_type;
	double trigger_delay_time;
	int trigger_pulses;

	double raster_grad;
	double raster_rf;
	double grad_max_ampl;
};


// for interactive mode
BARTLIB_API int BARTLIB_CALL
seq_check_equal_dims(int D, const long dims1[__VLA(D)], const long dims2[__VLA(D)], unsigned long flags);

struct seq_config;

BARTLIB_API extern int BARTLIB_CALL seq_raga_spokes(const struct seq_config* seq);

// minimum TE and TR calculation at end of prepare
BARTLIB_API extern double BARTLIB_CALL seq_minimum_tr(const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_minimum_te(const struct seq_config* seq, double* min_te, double* fill_te);


BARTLIB_API extern long BARTLIB_CALL seq_relevant_readouts_meas_time(const struct seq_config* seq);
BARTLIB_API extern double BARTLIB_CALL seq_total_measure_time(const struct seq_config* seq);


// conversion UI and seq_config
BARTLIB_API extern void BARTLIB_CALL
seq_ui_interface_loop_dims(int reverse, struct seq_config* seq, const int D, long dims[__VLA(D)]);

BARTLIB_API extern void BARTLIB_CALL
seq_ui_interface_custom_params(int reverse, struct seq_config* seq, int nl, long params_long[__VLA(nl)], int nd, double params_double[__VLA(nd)]);

BARTLIB_API extern void BARTLIB_CALL
seq_ui_interface_standard_conf(int reverse, struct seq_config* seq, struct seq_standard_conf* std_conf);

BARTLIB_API extern struct seq_interface_conf BARTLIB_CALL
seq_get_interface_conf(struct seq_config* seq);

BARTLIB_API extern void BARTLIB_CALL seq_set_fov_pos(int N, int M, const float* shifts, struct seq_config* seq);


BARTLIB_API extern int BARTLIB_CALL seq_print_info_config(int N, char* info, const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_print_info_radial_views(int N, char* info, const struct seq_config* seq);



BARTLIB_API int BARTLIB_CALL seq_config_from_string(struct seq_config* seq, int N, char* buffer);

#include "misc/cppwrap.h"

#endif // _SEQ_HELPERS_H
 

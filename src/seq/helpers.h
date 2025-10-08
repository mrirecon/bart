
#ifndef _SEQ_HELPERS_H
#define _SEQ_HELPERS_H

#include "misc/dllspec.h"
#include "misc/cppwrap.h"

#include "misc/mri.h"

#define MAX_PARAMS_LONG 64
#define MAX_PARAMS_DOUBLE 16

#define MAX_NO_ECHOES 64
#define MAX_SLICES 64
#define MAX_GRAD_POINTS 8192
#define ACOUSTIC_RES_ENTRIES 5


enum seq_order {

	SEQ_ORDER_AVG_OUTER,
	SEQ_ORDER_SEQ_MS,
	SEQ_ORDER_AVG_INNER,
};

enum context {

	CONTEXT_NORMAL,
	CONTEXT_BINARY,
	CONTEXT_UPDATE
};

enum mag_prep {

	PREP_OFF,
	PREP_IR_SEL,
	PREP_IR_NON,
	PREP_SR_SEL,
	PREP_SR_NON,
	PREP_SR_ADIAB
};


enum trigger_type {

	TRIGGER_OFF,
	TRIGGER_ECG,
	TRIGGER_PULSE,
	TRIGGER_RESP,
	TRIGGER_EXT,
};

struct seq_standard_conf {

	double tr;
	double te[MAX_NO_ECHOES];

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
	double acoustic_res_freq[ACOUSTIC_RES_ENTRIES];
	double acoustic_res_bw[ACOUSTIC_RES_ENTRIES];

	enum mag_prep mag_prep;
	double ti;

	enum trigger_type trigger_type;
	double trigger_delay_time;
	int trigger_pulses;
	int trigger_out;

	enum seq_order enc_order;

	enum context context;
};



// for interactive mode
BARTLIB_API int BARTLIB_CALL
seq_check_equal_dims(int D, const long dims1[__VLA(D)], const long dims2[__VLA(D)], unsigned long flags);

struct seq_config;

BARTLIB_API extern int BARTLIB_CALL seq_raga_spokes(const struct seq_config* seq);

// minimum TE and TR calculation at end of prepare
BARTLIB_API extern long BARTLIB_CALL seq_minimum_tr(const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_minimum_te(const struct seq_config* seq, long* min_te, long* fil_te);


BARTLIB_API extern long BARTLIB_CALL seq_relevant_readouts_meas_time(const struct seq_config* seq);
BARTLIB_API extern double BARTLIB_CALL seq_total_measure_time(const struct seq_config* seq);


// conversion UI and seq_config
BARTLIB_API extern void BARTLIB_CALL
seq_ui_interface_loop_dims(int reverse, struct seq_config* seq, const int D, long dims[__VLA(D)]);

BARTLIB_API extern void BARTLIB_CALL
seq_ui_interface_custom_params(int reverse, struct seq_config* seq, int nl, long params_long[__VLA(nl)], int nd, double params_double[__VLA(nd)]);

BARTLIB_API extern void BARTLIB_CALL
seq_ui_interface_standard_conf(int reverse, struct seq_config* seq, struct seq_standard_conf* std_conf);


BARTLIB_API extern long BARTLIB_CALL seq_get_slices(const struct seq_config* seq);
BARTLIB_API extern void BARTLIB_CALL seq_set_fov_pos(int N, int M, const float* shifts, struct seq_config* seq);



BARTLIB_API int BARTLIB_CALL seq_read_config_from_file(struct seq_config* seq, const char* filename);

#include "misc/cppwrap.h"

#endif // _SEQ_HELPERS_H

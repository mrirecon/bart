
#ifndef _SEQ_CONFIG_H
#define _SEQ_CONFIG_H

#include "misc/cppwrap.h"

#include "misc/mri.h"

#include "seq/gradient.h"
#include "seq/helpers.h"
#include "seq/seq.h"

#define SEQ_FLAGS (PHS1_FLAG|PHS2_FLAG|COEFF_FLAG|COEFF2_FLAG|TIME_FLAG|TIME2_FLAG|SLICE_FLAG|AVG_FLAG|BATCH_FLAG)


extern const int seq_loop_order_avg_inner[DIMS];
extern const int seq_loop_order_avg_outer[DIMS];
extern const int seq_loop_order_multislice[DIMS];
extern const int seq_loop_order_asl[DIMS];


enum flash_contrast {

	SEQ_CONTRAST_NO_SPOILING = 0,
	SEQ_CONTRAST_RF_RANDOM,
	SEQ_CONTRAST_RF_SPOILED
};


enum pe_mode {

	SEQ_PEMODE_TURN = 2,
	SEQ_PEMODE_RAGA,
	SEQ_PEMODE_MEMS_HYB,
};



struct seq_phys {

	double tr;
	double te;
	double te_delta;

	double dwell;
	double os;

	enum flash_contrast contrast;
	double rf_duration;
	double flip_angle; // deg
	double bwtp;
};

struct seq_geom {

	double fov;
	double slice_thickness;
	double shift[SEQ_MAX_SLICES][3]; // [ro, pe, slice]

	int baseres;

	int mb_factor;
	double sms_distance;
};

struct seq_enc {

	enum pe_mode pe_mode;
	int tiny;
	unsigned long aligned_flags;
	enum seq_order order;
};

struct seq_magn {

	enum mag_prep mag_prep;
	double ti;
	double init_delay;
	double inv_delay_time;
};

struct seq_sys {

	double gamma; // Hz/T
	double b0; // T
	struct grad_limits grad; // inv_slew_rate in s / (T/m), max_amplitude in T/m
	double coil_control_lead;
	double min_duration_ro_rf;
	double raster_grad;
	double raster_rf;
	double raster_dwell;
};


struct seq_trigger {

	// Physiological triggering: scanner waits on event
	enum trigger_type type; 
	double delay_time;
	int pulses;

	// Trigger output: scanner notifies other hardware
	int trigger_out;
};



struct seq_config {

	struct seq_phys phys;
	struct seq_geom geom;
	struct seq_enc enc;
	struct seq_magn magn;
	struct seq_trigger trigger;
	struct seq_sys sys; 

	int order[DIMS];
	long loop_dims[DIMS];
};

extern const struct seq_config seq_config_defaults;

#include "misc/cppwrap.h"

#endif	// _SEQ_CONFIG_H


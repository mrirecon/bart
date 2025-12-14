
#ifndef _SEQ_CONFIG_H
#define _SEQ_CONFIG_H

#include "misc/dllspec.h"
#include "misc/cppwrap.h"
#include "misc/mri.h"

#include "seq/gradient.h"

#define MAX_RF_PULSES 32
#define MAX_SLICES 64
#define MAX_GRAD_POINTS 8192

#define SEQ_FLAGS (PHS1_FLAG|PHS2_FLAG|COEFF_FLAG|COEFF2_FLAG|TIME_FLAG|TIME2_FLAG|SLICE_FLAG|AVG_FLAG|BATCH_FLAG)


extern const int seq_loop_order_avg_inner[DIMS];
extern const int seq_loop_order_avg_outer[DIMS];
extern const int seq_loop_order_multislice[DIMS];

enum seq_order {

	SEQ_ORDER_AVG_OUTER,
	SEQ_ORDER_SEQ_MS,
	SEQ_ORDER_AVG_INNER,
};

enum block {

	BLOCK_UNDEFINED,
	BLOCK_PRE,
	BLOCK_KERNEL_CHECK,
	BLOCK_KERNEL_PREPARE,
	BLOCK_KERNEL_NOISE,
	BLOCK_KERNEL_DUMMY ,
	BLOCK_KERNEL_IMAGE,
	BLOCK_POST 
};

enum flash_contrast {

	CONTRAST_NO_SPOILING = 0,
	CONTRAST_RF_RANDOM,
	CONTRAST_RF_SPOILED
};

enum mag_prep {

	PREP_OFF,
	PREP_IR_NON,
};


enum pe_mode {

	PEMODE_TURN = 2,
	PEMODE_MEMS_HYB,
	PEMODE_RAGA,
	PEMODE_RAGA_ALIGNED,
	PEMODE_RAGA_MEMS
};



struct seq_phys {

	double tr;
	double te;

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
	double shift[MAX_SLICES][3]; // [ro, pe, slice]

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



struct seq_config {

	struct seq_phys phys;
	struct seq_geom geom;
	struct seq_enc enc;
	struct seq_magn magn;
	struct seq_sys sys; 

	int order[DIMS];
	long loop_dims[DIMS];
};

struct seq_state {

	enum block mode;
	long chrono_slice;
	long pos[DIMS];
	double start_block;
};


BARTLIB_API extern const struct seq_config seq_config_defaults;

#include "misc/cppwrap.h"

#endif	// _SEQ_CONFIG_H


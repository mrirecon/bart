
#include <stdbool.h>

#include "simu/signals.h"

struct opt_reg_s;

enum mdb_t { MDB_T1, MDB_T2, MDB_MGRE };
enum edge_filter_t { EF1, EF2 };

#ifndef _MECO_MODEL
#define _MECO_MODEL 1
enum meco_model {
	MECO_WF,
	MECO_WFR2S,
	MECO_WF2R2S,
	MECO_R2S,
	MECO_PHASEDIFF,
	MECO_PI,
};
#endif

struct moba_conf {

	enum mdb_t mode;

	unsigned int iter;
	unsigned int opt_reg;
	float alpha;
	float alpha_min;
	bool alpha_min_exp_decay;
	float redu;
	float step;
	float lower_bound;
	float tolerance;
	float damping;
	unsigned int inner_iter;
	float sobolev_a;
	float sobolev_b;
	bool noncartesian;
        bool sms;

	bool k_filter;
	enum edge_filter_t k_filter_type;

	bool auto_norm_off;
	bool stack_frames;
	int algo;	// enum algo_t
	float rho;
	struct opt_reg_s* ropts;

	// MECO
	enum meco_model mgre_model;	// enum
	enum fat_spec fat_spec;
	float scale_fB0[2]; // { spatial smoothness, scaling }
	bool out_origin_maps;

	bool use_gpu;
};

extern struct moba_conf moba_defaults;



#include <stdbool.h>

#include "simu/signals.h"
#include "simu/pulse.h"
#include "simu/simulation.h"

struct opt_reg_s;

enum mdb_t { MDB_T1, MDB_T2, MDB_MGRE, MDB_T1_PHY, MDB_BLOCH };
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

	// T1
	float scaling_M0;
	float scaling_R1s;

	bool k_filter;
	enum edge_filter_t k_filter_type;

	bool auto_norm;
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
	int num_gpu;
};

extern struct moba_conf moba_defaults;

struct moba_other_conf {

        float fov_reduction_factor;
        float scale[4];
	float initval[4];

	int tvscales_N;
	complex float tvscales[4];
};

extern struct moba_other_conf moba_other_defaults;

struct sim_data;

struct moba_conf_s {

        enum mdb_t model;

        struct sim_data sim;
        struct moba_other_conf other;
};

extern int moba_get_nr_of_coeffs(const struct moba_conf* conf, int in);

extern void debug_other(struct moba_other_conf* data);


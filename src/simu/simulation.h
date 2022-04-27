
#include <complex.h>
#include <stdbool.h>

#ifndef SIMULATION_H
#define SIMULATION_H

typedef enum sim_seq_t {BSSFP, IRBSSFP, FLASH, IRFLASH} sim_seq;

struct simdata_voxel {

	float r1;
	float r2;
	float m0;
	float w;
	float b1;
};
extern const struct simdata_voxel simdata_voxel_defaults;


struct simdata_seq {

	sim_seq seq_type;
	float tr;
	float te;
	int rep_num;
	int spin_num;

	bool perfect_inversion;
	float inversion_pulse_length;
        float inversion_spoiler;

	float prep_pulse_length;

        bool pulse_applied;
};
extern const struct simdata_seq simdata_seq_defaults;


struct simdata_tmp {

        int rep_counter;
	int spin_counter;
	float t;
	float w1;
	float r2spoil;
};
extern const struct simdata_tmp simdata_tmp_defaults;


struct simdata_grad {

	float gb[3];
	float gb_eff[3];
	float mom;
	float mom_sl;
};
extern const struct simdata_grad simdata_grad_defaults;

struct simdata_pulse;

struct sim_data {

	struct simdata_seq seq;
	struct simdata_voxel voxel;
	struct simdata_pulse pulse;
	struct simdata_grad grad;
	struct simdata_tmp tmp;
};

extern void debug_sim(struct sim_data* data);
extern void start_rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N]);
extern void inversion(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end);
extern void bloch_simulation(struct sim_data* data, float (*mxy_sig)[3], float (*sa_r1_sig)[3], float (*sa_r2_sig)[3], float (*sa_m0_sig)[3], float (*sa_b1_sig)[3]);

#endif
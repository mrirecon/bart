
#ifndef SIMULATION_H
#define SIMULATION_H

#include <complex.h>
#include <stdbool.h>


enum sim_seq { SEQ_BSSFP, SEQ_IRBSSFP, SEQ_FLASH, SEQ_IRFLASH };
enum sim_type { SIM_ROT, SIM_ODE, SIM_STM };

struct simdata_voxel {

	float r1;
	float r2;
	float m0;
	float w;
	float b1;
};

extern const struct simdata_voxel simdata_voxel_defaults;


struct simdata_seq {

        enum sim_type type;
	enum sim_seq seq_type;
	float tr;
	float te;
	int rep_num;
	int spin_num;

	bool perfect_inversion;
	float inversion_pulse_length;
        float inversion_spoiler;

	float prep_pulse_length;

        int averaged_spokes;
	float slice_thickness;
	float nom_slice_thickness;

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
	float sl_gradient_strength;
	float mom;
	float mom_sl;
};

extern const struct simdata_grad simdata_grad_defaults;


struct simdata_other {

	float ode_tol;
        float sampling_rate;
};

extern const struct simdata_other simdata_other_defaults;


struct simdata_pulse;

struct sim_data {

	struct simdata_seq seq;
	struct simdata_voxel voxel;
	struct simdata_pulse pulse;
	struct simdata_grad grad;
	struct simdata_tmp tmp;
	struct simdata_other other;
};

extern void debug_sim(struct sim_data* data);
extern void rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float stm_matrix[P*N + 1][P*N + 1]);

extern void inversion(const struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end);
extern void bloch_simulation(const struct sim_data* data, int R, float (*m_state)[R][3], float (*sa_r1_state)[R][3], float (*sa_r2_state)[R][3], float (*sa_m0_state)[R][3], float (*sa_b1_state)[R][3]);

struct ode_matrix_simu_s {

	int N;
	struct sim_data* sim_data;
};

extern void ode_matrix_interval_simu(struct sim_data* _data, float h, float tol, unsigned int N, float out[N], float st, float end);

extern void mat_exp_simu(struct sim_data* data, int N, float st, float end, float out[N][N]);
extern void apply_sim_matrix(int N, float m[N], float matrix[N][N]);

#endif

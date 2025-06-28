
#ifndef _SIMULATION_H
#define _SIMULATION_H

#include <complex.h>
#include <stdbool.h>


enum sim_seq { SEQ_BSSFP, SEQ_IRBSSFP, SEQ_FLASH, SEQ_IRFLASH, SEQ_CEST };
enum sim_type { SIM_ROT, SIM_ODE, SIM_STM };
enum sim_model { MODEL_BLOCH, MODEL_BMC };

struct simdata_voxel {

	int P;

	float r1[5];
	float r2[5];
	float m0[5];
	float Om[5];

	float k[4];

	float w;
	float b1;
};

extern const struct simdata_voxel simdata_voxel_defaults;


struct simdata_seq {

        enum sim_type type;
	enum sim_seq seq_type;
	enum sim_model model;
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


struct simdata_grad {

	float gb[3];
	float sl_gradient_strength;
	float mom;
	float mom_sl;
};

extern const struct simdata_grad simdata_grad_defaults;


struct simdata_other {

	float ode_tol;
	float stm_tol;
        float sampling_rate;
};

extern const struct simdata_other simdata_other_defaults;

enum pulse_t { PULSE_SINC, PULSE_HS, PULSE_REC };

struct pulse_sinc;
struct pulse_hypsec;
struct pulse_rect;

struct simdata_pulse {

	enum pulse_t type;

	float rf_start;
	float rf_end;

	float phase;

	struct pulse_sinc sinc;
	struct pulse_rect rect;
        struct pulse_hypsec hs;
};

struct simdata_cest {

	int n_pulses;
	float t_d;
	float t_pp;
	float gamma;
	float b1_amp;
	float b0;

	float off_start;
	float off_stop;

	bool ref_scan;
	float ref_scan_ppm;
};

extern const struct simdata_cest simdata_cest_defaults;

extern void pulse_init(struct simdata_pulse* pulse, float rf_start, float rf_end, float angle, float phase, float bwtp, float alpha);
extern const struct simdata_pulse simdata_pulse_defaults;


struct sim_data {

	struct simdata_seq seq;
	struct simdata_voxel voxel;
	struct simdata_pulse pulse;
	struct simdata_grad grad;
	struct simdata_other other;
	struct simdata_cest cest;
};

extern void debug_sim(struct sim_data* data);

extern void rf_pulse(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float stm_matrix[P * N][P * N]);
extern void relaxation2(struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end, float stm_matrix[P * N][P * N], float r2spoil);

extern void inversion(const struct sim_data* data, float h, float tol, int N, int P, float xp[P][N], float st, float end);
extern void bloch_simulation(const struct sim_data* _data, int R, float (*m_state)[R][3], float (*sa_r1_state)[R][3], float (*sa_r2_state)[R][3], float (*sa_m0_state)[R][3],	float (*sa_b1_state)[R][3]);
extern void bloch_simulation2(const struct sim_data* _data, int R, int pools, float (*m_state)[R][pools][3], float (*sa_r1_state)[R][pools][3], float (*sa_r2_state)[R][pools][3], float (*sa_m0_state)[R][pools][3], float (*sa_b1_state)[R][1][3], float (*sa_k_state)[R][pools][3], float (*sa_om_state)[R][pools][3]);

extern void mat_exp_simu(struct sim_data* data, float r2spoil, int N, float st, float end, float out[N][N]);
extern void apply_sim_matrix(int N, float m[N], float matrix[N][N]);

#endif

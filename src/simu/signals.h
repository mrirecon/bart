
#ifndef _SIGNALS_H
#define _SIGNALS_H 1

#include <complex.h>
#include <stdbool.h>


enum fat_spec {

	FAT_SPEC_0,
	FAT_SPEC_1,
};

struct signal_model {
	
	float m0;
	float m0_water;
	float m0_fat;
	float ms;
	float t1;
	float t2;
	float t2star;
	float te;
	float tr;
	float b0;
	float off_reson;
	float fa;
	float beta;
	bool ir;
	bool ir_ss;
	enum fat_spec fat_spec;
        float time_T1relax;
        long Hbeats;

        int averaged_spokes;
};


extern const struct signal_model signal_TSE_defaults;

extern void TSE_model(const struct signal_model* data, int N, complex float out[N]);


extern const struct signal_model signal_hsfp_defaults;

extern void hsfp_simu(const struct signal_model* data, int N, const float pa[N], complex float out[N]);


extern const struct signal_model signal_looklocker_defaults;

extern void looklocker_model(const struct signal_model* data, int N, complex float out[N]);

extern void looklocker_model2(const struct signal_model* data, int N, complex float out[N]);

extern void MOLLI_model(const struct signal_model* data, int N, complex float out[N]);


extern const struct signal_model signal_IR_bSSFP_defaults;

extern void IR_bSSFP_model(const struct signal_model* data, int N, complex float out[N]);


extern const struct signal_model signal_multi_grad_echo_defaults;
extern const struct signal_model signal_multi_grad_echo_fat;

extern complex float calc_fat_modulation(float b0, float TE, enum fat_spec fs);

extern void multi_grad_echo_model(const struct signal_model* data, int N, complex float out[N]);


#endif // _SIGNALS_H


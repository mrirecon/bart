
#include <complex.h>

extern void create_relax_matrix_der(complex float ee[3][3], complex float dee[4][3][3], float T1, float T2, float offres, float TE);
extern void create_relax_matrix(complex float ee[3][3], float T1, float T2, float offres,  float TE);
extern void create_rf_pulse_matrix_der(complex float T[3][3], complex float dT[4][3][3], float alpha, float phi);
extern void create_rf_pulse_matrix(complex float T[3][3], float alpha, float phi);

extern void epg_pulse(complex float T[3][3], int num_cols, complex float states[3][num_cols]);
extern void epg_pulse_der(complex float T[3][3], int num_cols, complex float states[3][num_cols],
        complex float dT[4][3][3], complex float dstates[4][3][num_cols]);

extern void epg_relax_der(complex float ee[3][3], int num_cols, complex float states[3][num_cols],
		complex float dee[4][3][3], complex float dstates[4][3][num_cols]);
extern void epg_relax(complex float ee[3][3], int num_cols, complex float states[3][num_cols]);

extern void epg_grad_der(int num_cols, complex float states[3][num_cols],complex float dstates[4][3][num_cols]);
extern void epg_grad(int num_cols, complex float states[3][num_cols]);

extern void epg_spoil_der(int num_cols, complex float states[3][num_cols], complex float dstates[4][3][num_cols]);
extern void epg_spoil(int num_cols, complex float states[3][num_cols]);

extern void epg_adc_der(int idx_current, int N, int M, complex float signal[N],
		 complex float states[3][M][N], complex float dsignal[4][N],
		 complex float dstates[4][3][M][N],
		 complex float state_current[3][M],
		 complex float dstate_current[4][3][M],
		 float adc_phase);
extern void epg_adc(int idx_current, int N, int M, complex float signal[N],
		 complex float states[3][M][N],
		 complex float state_current[3][M],
		 float adc_phase);

extern void cpmg_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N],
        complex float dsignal[4][N], complex float dstates[4][3][M][N], float rf_exc,
        float rf_ref, float TE, float T1, float T2, float B1, float offres);

extern void fmssfp_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N],
        complex float dsignal[4][N], complex float dstates[4][3][M][N],
        float FA, float TR, float T1, float T2, float B1, float offres);
extern void fmssfp_epg(int N, int M, complex float signal[N], complex float states[3][M][N],
        float FA, float TR, float T1, float T2, float B1, float offres);

extern void hyperecho_epg(int N, int M, complex float signal[N], complex float states[3][M][N],
        float rf_exc, float rf_ref, float TE, float FA, float T1, float T2, float B1, float offres);

extern void flash_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N],
        complex float dsignal[4][N], complex float dstates[4][3][M][N],
        float FA, float TR, float T1, float T2, float B1, float offres, int spoiling);

extern void hahnspinecho_epg(int N, int M, complex float signal[N], complex float states[3][M][N],
        float FA, float TE, float T1, float T2, float B1, float offres);

extern void bssfp_epg_der(int N, int M, complex float signal[N], complex float states[3][M][N], complex float dsignal[4][N], complex float dstates[4][3][M][N], float FA, float TR, float T1, float T2, float B1, float offres);

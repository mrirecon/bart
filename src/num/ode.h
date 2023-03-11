
#include "misc/nested.h"

extern void rk4_step(float h, int N, float ynp[N], float tn, const float yn[N],
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn));

extern void dormand_prince_step(float h, int N, float ynp[N], float tn, const float yn[N],
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn));

extern float dormand_prince_step2(float h, int N, float ynp[N], float tn, const float yn[N], float tmp[6][N],
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn));

extern float dormand_prince_scale(float tol, float err);

extern void ode_interval(float h, float tol, int N, float x[N], float st, float end,
		void CLOSURE_TYPE(f)(float* out, float t, const float* yn));

extern void ode_matrix_interval(float h, float tol, int N, float x[N], float st, float end, const float matrix[N][N]);

extern void ode_direct_sa(float h, float tol, int N, int P, float x[P + 1][N],
	float st, float end,
	void CLOSURE_TYPE(f)(float* out, float t, const float* yn),
	void CLOSURE_TYPE(pdy)(float* out, float t, const float* yn),
	void CLOSURE_TYPE(pdp)(float* out, float t, const float* yn));

extern void ode_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M],
	void CLOSURE_TYPE(sys)(float dst[M], float t, const float in[M]),
	void CLOSURE_TYPE(sysT)(float dst[M], float t, const float in[M]),
	void CLOSURE_TYPE(cost)(float dst[M], float t));

extern void ode_matrix_adjoint_sa(float h, float tol,
	int N, const float t[N + 1],
	int M, float x[N + 1][M], float z[N + 1][M],
	const float x0[M], const float sys[N][M][M],
	const float cost[N][M]);
#if 0
extern void ode_matrix_adjoint_sa(int N, const float t[N + 1],
	int M, float z[N + 1][M], const float x0[M],
	void (*sys)(float sys[M][M], float t),
	void (*cost)(float c[N], float t));
#endif

extern void ode_adjoint_sa_eval(int N, const float t[N + 1], int M,
		int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M]);

void ode_adjoint_sa_eq_eval(int N, int M, int P, float dj[P],
		const float x[N + 1][M], const float z[N + 1][M],
		const float Adp[P][M][M]);


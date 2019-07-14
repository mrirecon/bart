
extern void rk4_step(float h, unsigned int N, float ynp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn));

extern void dormand_prince_step(float h, unsigned int N, float ynp[N], float tn, const float yn[N], void* data, void (*f)(void* data, float* out, float t, const float* yn));

extern float dormand_prince_step2(float h, unsigned int N, float ynp[N], float tn, const float yn[N], float tmp[6][N], void* data, void (*f)(void* data, float* out, float t, const float* yn));

extern float dormand_prince_scale(float tol, float err);

extern void ode_interval(float h, float tol, unsigned int N, float x[N], float st, float end, void* data, void (*f)(void* data, float* out, float t, const float* yn));

extern void ode_matrix_interval(float h, float tol, unsigned int N, float x[N], float st, float end, const float matrix[N][N]);

extern void ode_direct_sa(float h, float tol, unsigned int N, unsigned int P, float x[P + 1][N],
	float st, float end, void* data,
	void (*f)(void* data, float* out, float t, const float* yn),
	void (*pdy)(void* data, float* out, float t, const float* yn),
	void (*pdp)(void* data, float* out, float t, const float* yn));


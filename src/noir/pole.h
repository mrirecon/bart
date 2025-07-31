
struct lseg_s {

	int N;
	float (*pos)[2][3];
};

struct pole_config_s {

	float diameter;
	float closing;
	int segments;
	float thresh;
	unsigned long avg_flag;
	int normal;
};

extern struct pole_config_s pole_config_default;

extern void compute_curl_map(struct pole_config_s conf, int N, const long curl_dims[N], int dim, _Complex float* curl_map, const long sens_dims[N], const _Complex float* sens);
extern void compute_curl_weighting(struct pole_config_s conf, int N, const long curl_dims[N], int dim, _Complex float* wgh_map, const long cim_dims[N], const _Complex float* cim);
extern void average_curl_map(int N, const long pmap_dims[N], _Complex float* red_curl_map, const long curl_dims[N], int dim, _Complex float* curl_map, _Complex float* wgh_map);

extern void sample_phase_pole_2D(int N, const long dims[N], _Complex float* dst, int D, const float pos[D][2][3]);
struct lseg_s extract_phase_poles_2D(struct pole_config_s conf, int N, const long dims[N], const _Complex float* curl_map);

extern _Bool phase_pole_correction(struct pole_config_s conf, int N, const long pmap_dims[N], _Complex float* phase, const long sens_dims[N], const _Complex float* sens);
extern _Bool phase_pole_correction_loop(struct pole_config_s conf, int N, unsigned long lflags, const long pmap_dims[N], _Complex float* phase, const long sens_dims[N], const _Complex float* sens);

extern void phase_pole_normalize(int N, const long pdims[N], _Complex float* phase, const long idims[N], const _Complex float* image);




struct iter3_irgnm_conf;
struct nlop_s;
struct opt_reg_s;


#ifndef DIMS
#define DIMS 16
#endif

struct mdb_irgnm_l1_conf {

	struct iter3_irgnm_conf* c2;
	unsigned int opt_reg;

	float step;
	float lower_bound;
	int constrained_maps;
	bool auto_norm;

	int not_wav_maps;
	unsigned int algo;
	float rho;
	struct opt_reg_s* ropts;
};

void mdb_irgnm_l1(const struct mdb_irgnm_l1_conf* conf,
		const long dims[DIMS],
		struct nlop_s* nlop,
		long N, float* dst,
		long M, const float* src);


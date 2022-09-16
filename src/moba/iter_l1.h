

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
	unsigned long constrained_maps;
	unsigned long l2flags;
	_Bool auto_norm;

	int not_wav_maps;
	unsigned int algo;
	float rho;
	struct opt_reg_s* ropts;
	int tvscales_N;
	complex float* tvscales;
};

void mdb_irgnm_l1(const struct mdb_irgnm_l1_conf* conf,
		const long dims[DIMS],
		struct nlop_s* nlop,
		long N, float* dst,
		long M, const float* src);


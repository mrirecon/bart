
#include <stdbool.h>

struct opt_reg_s;

struct moba_conf {

	unsigned int iter;
	unsigned int opt_reg;
	float alpha;
	float alpha_min;
	bool alpha_min_exp_decay;
	float redu;
	float step;
	float lower_bound;
	float tolerance;
	float damping;
	unsigned int inner_iter;
	float sobolev_a;
	float sobolev_b;
	bool noncartesian;
        bool sms;
	bool k_filter;
	bool auto_norm_off;
	bool stack_frames;
	int algo;	// enum algo_t
	float rho;
	struct opt_reg_s* ropts;
};



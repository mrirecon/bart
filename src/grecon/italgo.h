
struct admm_conf {

	bool dynamic_rho;
	bool dynamic_tau;
	bool relative_norm;
	float rho;
	unsigned int maxitercg;
};


struct iter {

	italgo_fun2_t italgo;
	iter_conf* iconf;
};

struct reg_s;
enum algo_t;

extern struct iter configure_italgo(enum algo_t algo, int nr_penalties, const struct reg_s* regs, unsigned int maxiter, float step, bool hogwild, bool fast, const struct admm_conf admm, float scaling, bool warm_start);

extern void configure_italgo_free(struct iter it);



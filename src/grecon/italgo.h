
enum algo_t { ALGO_DEFAULT, ALGO_CG, ALGO_IST, ALGO_FISTA, ALGO_ADMM, ALGO_NIHT, ALGO_PRIDU };

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

extern enum algo_t italgo_choose(int nr_penalties, const struct reg_s regs[nr_penalties]);

extern struct iter italgo_config(enum algo_t algo, int nr_penalties, const struct reg_s* regs, unsigned int maxiter, float step, bool hogwild, bool fast, const struct admm_conf admm, float scaling, bool warm_start);

extern void italgo_config_free(struct iter it);



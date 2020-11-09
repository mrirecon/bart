
#include <stdbool.h>

struct moba_conf {

	unsigned int iter;
	unsigned int opt_reg;
	float alpha;
	float alpha_min;
	float redu;
	float step;
	float lower_bound;
	float tolerance;
	unsigned int inner_iter;
	bool noncartesian;
        bool sms;
	bool k_filter;
	bool auto_norm_off;
};




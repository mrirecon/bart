
#include "misc/types.h"

typedef struct iter3_conf_s { TYPEID* TYPEID; } iter3_conf;

struct iter_op_s;



struct iter3_irgnm_conf {

	iter3_conf super;

	int iter;
	float alpha;
	float alpha_min;
	float alpha_min0;
	float redu;

	int cgiter;
	float cgtol;

	_Bool nlinv_legacy;
};

struct iter3_lbfgs_conf {

	iter3_conf super;

	int iter;
	int M;
	float step;
	float c1;
	float c2;
	float ftol;
	float gtol;
};


struct iter3_levenberg_marquardt_conf {

	INTERFACE(iter3_conf);

	int iter;
	int cgiter;
	float redu;
	long Bi;
	long Bo;
	float l2lambda;
};




struct iter3_landweber_conf {

	iter3_conf super;

	int iter;
	float alpha;
	float epsilon;
};




extern const struct iter3_irgnm_conf iter3_irgnm_defaults;
extern const struct iter3_landweber_conf iter3_landweber_defaults;
extern const struct iter3_lbfgs_conf iter3_lbfgs_defaults;
extern const struct iter3_levenberg_marquardt_conf iter3_levenberg_marquardt_defaults;


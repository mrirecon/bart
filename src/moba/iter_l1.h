

struct iter3_conf_s;
struct nlop_s;


#ifndef DIMS
#define DIMS 16
#endif

void mdb_irgnm_l1(const struct iter3_conf_s* _conf,
		const long dims[DIMS],
		struct nlop_s* nlop,
		long N, float* dst,
		long M, const float* src);


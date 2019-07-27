


struct iter3_conf_s;
struct iter_nlop_s;
struct nlop_s;

typedef void iter5_altmin_f(iter3_conf* _conf,
			struct nlop_s* nlop,
			long NI, float* dst[NI],
			long M, const float* src,
			struct iter_nlop_s cb);

iter5_altmin_f iter5_altmin;



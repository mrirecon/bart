
extern const struct nlop_s* nlop_zaxpbz_create(int N, const long dims[__VLA(N)], _Complex float scale1, _Complex float scale2);
extern const struct nlop_s* nlop_smo_abs_create(int N, const long dims[__VLA(N)], float epsilon);

extern const struct nlop_s* nlop_zmax_create(int N, const long dims[__VLA(N)], unsigned long flags);

extern const struct nlop_s* nlop_dump_create(int N, const long dims[__VLA(N)], const char* filename, _Bool frw, _Bool der, _Bool adj);
extern const struct nlop_s* nlop_zinv_create(int N, const long dims[__VLA(N)]);
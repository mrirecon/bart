
extern const struct nlop_s* nlop_zaxpbz2_create(int N, const long dims[__VLA(N)], unsigned long flags1, _Complex float scale1, unsigned long flags2, _Complex float scale2);
extern const struct nlop_s* nlop_zaxpbz_create(int N, const long dims[__VLA(N)], _Complex float scale1, _Complex float scale2);

extern const struct nlop_s* nlop_zsadd_create(int N, const long dims[__VLA(N)], _Complex float val);
extern const struct nlop_s* nlop_smo_abs_create(int N, const long dims[__VLA(N)], float epsilon);

extern const struct nlop_s* nlop_zmax_create(int N, const long dims[__VLA(N)], unsigned long flags);

extern const struct nlop_s* nlop_dump_create(int N, const long dims[__VLA(N)], const char* filename, _Bool frw, _Bool der, _Bool adj);

extern const struct nlop_s* nlop_zinv_reg_create(int N, const long dims[__VLA(N)], float eps);
extern const struct nlop_s* nlop_zinv_create(int N, const long dims[__VLA(N)]);

extern const struct nlop_s* nlop_zdiv_reg_create(int N, const long dims[__VLA(N)], float eps);
extern const struct nlop_s* nlop_zdiv_create(int N, const long dims[__VLA(N)]);

extern const struct nlop_s* nlop_zsqrt_create(int N, const long dims[__VLA(N)]);
extern const struct nlop_s* nlop_zspow_create(int N, const long dims[__VLA(N)], _Complex float exp);

extern const struct nlop_s* nlop_zss_create(int N, const long dims[__VLA(N)], unsigned long flags);

extern const struct nlop_s* nlop_zrss_reg_create(int N, const long dims[__VLA(N)], unsigned long flags, float epsilon);
extern const struct nlop_s* nlop_zrss_create(int N, const long dims[__VLA(N)], unsigned long flags);

extern const struct nlop_s* nlop_zabs_create(int N, const long dims[__VLA(N)]);
extern const struct nlop_s* nlop_zphsr_create(int N, const long dims[__VLA(N)]);

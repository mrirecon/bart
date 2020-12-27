
struct linop_s;
extern const struct linop_s* linop_avg_create(int N, const long imgd_dims[N], unsigned long flags);
extern const struct linop_s* linop_sum_create(int N, const long imgd_dims[N], unsigned long flags);


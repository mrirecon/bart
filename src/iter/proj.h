
const struct operator_p_s* operator_project_pos_real_create(long N, const long dims[N]);
const struct operator_p_s* operator_project_mean_free_create(long N, const long dims[N], unsigned long bflag);
const struct operator_p_s* operator_project_sphere_create(long N, const long dims[N], unsigned long bflag, bool real);
const struct operator_p_s* operator_project_mean_free_sphere_create(long N, const long dims[N], unsigned long bflag, bool real);
const struct operator_p_s* operator_project_min_real_create(long N, const long dims[N], float min);
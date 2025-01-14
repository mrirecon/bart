struct nlop_s;

extern const struct nlop_s* nlop_lorentzian_multi_pool_create(int N, const long signal_dims[N],
		const long param_dims[N], const long omega_dims[N], const _Complex float* omega);

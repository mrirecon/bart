
struct linop_s;

extern const struct linop_s* linop_casorati_create(int N, const long kdim[__VLA(N)], const long ddims[__VLA(N)], const _Complex float* data);
extern const struct linop_s* linop_casoratiH_create(int N, const long kdim[__VLA(N)], const long ddims[__VLA(N)], const _Complex float* data);

extern void casorati_gram(int M, _Complex float out[M][M], int N, const long kdims[__VLA(N)], const long dims[__VLA(N)], const _Complex float* data);

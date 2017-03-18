
typedef float vec3_t[3];
extern void biot_savart(vec3_t x, const vec3_t r, unsigned int N, const vec3_t curve[static N]);

extern void vec3_ring(unsigned int N, vec3_t ring[N], const vec3_t c, const vec3_t n, float r);



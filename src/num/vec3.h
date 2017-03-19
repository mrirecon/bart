
typedef float vec3_t[3];


extern void vec3_saxpy(vec3_t dst, const vec3_t src1, float alpha, const vec3_t src2);
extern void vec3_sub(vec3_t dst, const vec3_t src1, const vec3_t src2);
extern void vec3_add(vec3_t dst, const vec3_t src1, const vec3_t src2);
extern void vec3_copy(vec3_t dst, const vec3_t src);
extern void vec3_clear(vec3_t dst);
extern float vec3_sdot(const vec3_t a, const vec3_t b);
extern float vec3_norm(const vec3_t x);
extern void vec3_rot(vec3_t dst, const vec3_t src1, const vec3_t src2);
extern void vec3_smul(vec3_t dst, const vec3_t src, float alpha);




typedef float vec3_t[3];

extern float triangle_intersect(float uv[2], const vec3_t o, const vec3_t d, const vec3_t tri[3]);
extern _Bool triangle2d_inside(const float tri[3][2], const float p[2]);


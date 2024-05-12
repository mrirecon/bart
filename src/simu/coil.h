
struct coil {

	float size;
	float dist;
};

extern const struct coil coil_defaults;

extern complex float coil(const struct coil* data, float x[3], int N, int i);


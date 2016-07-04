
struct na_s;
extern struct na_s* na_load(const char* name);
extern na na_create(const char* name, unsigned int N, const long (*dims)[N], size_t size);


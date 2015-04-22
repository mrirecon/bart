
extern void md_shuffle2(unsigned int N, const long dims[N], const long factors[N],
		const long ostrs[N], void* out, const long istrs[N], const void* in, size_t size);

extern void md_shuffle(unsigned int N, const long dims[N], const long factors[N],
		void* out, const void* in, size_t size);

extern void md_decompose2(unsigned int N, const long factors[N],
		const long odims[N + 1], const long ostrs[N + 1], void* out,
		const long idims[N], const long istrs[N], const void* in, size_t size);

extern void md_decompose(unsigned int N, const long factors[N], const long odims[N + 1], 
		void* out, const long idims[N], const void* in, size_t size);

extern void md_recompose2(unsigned int N, const long factors[N],
		const long odims[N], const long ostrs[N], void* out,
		const long idims[N + 1], const long istrs[N + 1], const void* in, size_t size);

extern void md_recompose(unsigned int N, const long factors[N], const long odims[N], 
		void* out, const long idims[N + 1], const void* in, size_t size);


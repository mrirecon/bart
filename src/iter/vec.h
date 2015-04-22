
#ifndef __ITER_VEC_H
#define __ITER_VEC_H

struct vec_iter_s {

	float* (*allocate)(long N);
	void (*del)(float* x);
	void (*clear)(long N, float* x);
	void (*copy)(long N, float* a, const float* x);
	void (*swap)(long N, float* a, float* x);

	double (*norm)(long N, const float* x);
	double (*dot)(long N, const float* x, const float* y);

	void (*sub)(long N, float* a, const float* x, const float* y);
	void (*add)(long N, float* a, const float* x, const float* y);

	void (*smul)(long N, float alpha, float* a, const float* x);
	void (*xpay)(long N, float alpha, float* a, const float* x);
	void (*axpy)(long N, float* a, float alpha, const float* x);
};

#ifdef USE_CUDA
extern const struct vec_iter_s gpu_iter_ops;
#endif
extern const struct vec_iter_s cpu_iter_ops;

extern const struct vec_iter_s* select_vecops(const float* x);


#endif


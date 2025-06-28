
#ifndef _ITER_VEC_H
#define _ITER_VEC_H

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
	void (*axpbz)(long N, float* out, const float a, const float* x, const float b, const float* z);
	void (*fmac)(long N, float* a, const float* x, const float* y);

	void (*mul)(long N, float* a, const float* x, const float* y);
	void (*div)(long N, float* a, const float* x, const float* y);
	void (*sqrt)(long N, float* a, const float* x);

	void (*smax)(long N, float alpha, float* a, const float* x);
	void (*smin)(long N, float alpha, float* a, const float* x);
	void (*sadd)(long N, float* x, float y);
	void (*sdiv)(long N, float* a, float x, const float* y);
	void (*le)(long N, float* a, const float* x, const float* y);

	void (*zmul)(long N, _Complex float* dst, const _Complex float* src1, const _Complex float* src2);
	void (*zsmax)(long N, float val, _Complex float* dst, const _Complex float* src1);

	void (*rand)(long N, float* dst);

	void (*xpay_bat)(long Bi, long N, long Bo, const float* beta, float* a, const float* x);
	void (*dot_bat)(long Bi, long N, long Bo, float* dst, const float* src1, const float* src2);
	void (*axpy_bat)(long Bi, long N, long Bo, float* a, const float* alpha, const float* x);
};

#ifdef USE_CUDA
extern const struct vec_iter_s gpu_iter_ops;
#endif
extern const struct vec_iter_s cpu_iter_ops;

extern const struct vec_iter_s* select_vecops(const float* x);


#endif


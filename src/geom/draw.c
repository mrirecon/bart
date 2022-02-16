/* Copyright 2019-2020. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2019-2020 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <stdbool.h>
#include <complex.h>
#include <stdlib.h>
#include <math.h>

#include "misc/nested.h"

#include "num/splines.h"

#include "draw.h"


#define MIN(a, b) ({ __auto_type __a = (a); __auto_type __b = (b); (__a < __b) ? __a : __b; })
#define MAX(a, b) ({ __auto_type __a = (a); __auto_type __b = (b); (__a > __b) ? __a : __b; })
#define SWAP(x, y)	({ __auto_type __t = (x); (x) = (y); (y) = __t; })


typedef void CLOSURE_TYPE(pixel_f)(int x, int y, float c);
typedef void line_f(pixel_f out, int x0, int y0, int x1, int y1);

static void setup(line_f line, int X, int Y, pixel_f out, int x0, int y0, int x1, int y1)
{
	if (   ((x0 < 0) && (x1 < 0))
	    || ((y0 < 0) && (y1 < 0))
	    || ((x0 >= X) && (x1 >= X))
	    || ((y0 >= Y) && (y1 >= Y)))
	    return;

	NESTED(void, out2, (int x2, int y2, float c))
	{
		if ((0 <= x2) && (x2 < X) && (0 <= y2) && (y2 < Y))
			out(x2, y2, c);
	};

	if (abs(x1 - x0) < abs(y1 - y0)) {

		NESTED(void, outT, (int x2, int y2, float c))
		{
			out2(y2, x2, c);
		};

		return line(outT, y0, x0, y1, x1);
	}

	return line(out2, x0, y0, x1, y1);
}

static void bresenham(pixel_f out, int x0, int y0, int x1, int y1)
{
	if (x1 < x0) {

		SWAP(x0, x1);
		SWAP(y0, y1);
	}

	int dx = x1 - x0;
	int dy = y1 - y0;
	int yi = 1;

	if (dy < 0) {

		dy *= -1;
		yi *= -1;
	}

	int D = 2 * dy - dx;
	int y = y0;

	for (int x = x0; x <= x1; x++) {

		out(x, y, 1.);

		if (D > 0) {

			y += yi;
			D -= 2 * dx;
		}

		D += 2 * dy;
	}
}



extern void bresenham_rgba_fl(int X, int Y, float (*out)[X][Y][4], const float (*val)[4], int x0, int y0, int x1, int y1)
{
	void* p = out;	// clang limitation

	NESTED(void, draw, (int x, int y, float c))
	{
		float (*out)[X][Y][4] = p;

		for (int i = 0; i < 4; i++)
			(*out)[x][y][i] = c * (*val)[i];
	};

	setup(bresenham, X, Y, draw, x0, y0, x1, y1);
}

extern void bresenham_rgba(int X, int Y, unsigned char (*out)[X][Y][4], const unsigned char (*val)[4], int x0, int y0, int x1, int y1)
{
	void* p = out;	// clang limitation

	NESTED(void, draw, (int x, int y, float c))
	{
		unsigned char (*out)[X][Y][4] = p;

		for (int i = 0; i < 4; i++)
			(*out)[x][y][i] = c * (*val)[i];
	};

	setup(bresenham, X, Y, draw, x0, y0, x1, y1);
}

extern void bresenham_cmplx(int X, int Y, complex float (*out)[X][Y], complex float val, int x0, int y0, int x1, int y1)
{
	void* p = out;	// clang limitation

	NESTED(void, draw, (int x, int y, float c))
	{
		complex float (*out)[X][Y] = p;

		(*out)[x][y] = c * val;
	};

	setup(bresenham, X, Y, draw, x0, y0, x1, y1);
}

static void xiaolin_wu(pixel_f out, int x0, int y0, int x1, int y1)
{
	if (x1 < x0) {

		SWAP(x0, x1);
		SWAP(y0, y1);
	}

	float dx = x1 - x0;
	float dy = y1 - y0;

	float grad = (0. == dx) ? 1. : (dy / dx);

	NESTED(float, frac, (float x)) { return x - floorf(x); };
	NESTED(float, rfrac, (float x)) { return (float)(1. - frac(x)); };

	NESTED(void, endp, (float x, float y))
	{
		float ye = y + grad * (roundf(x) - x);
		float gp = rfrac(x + 0.5);

		out(roundf(x), floorf(ye) + 0, gp * rfrac(ye));
		out(roundf(x), floorf(ye) + 1, gp * frac(ye));
	};

	endp(x0, y0);
	endp(x1, y1);

	float yi = y0 + grad * (1. + roundf(x0) - x0);

	for (float x = roundf(x0) + 1; x <= roundf(x1) - 1; x++, yi += grad) {
#if 1
		out(x, floorf(yi) + 0, rfrac(yi));
		out(x, floorf(yi) + 1, frac(yi));
#endif
	}
}


extern void xiaolin_wu_cmplx(int X, int Y, complex float (*out)[X][Y], complex float val, int x0, int y0, int x1, int y1)
{
	void* p = out;	// clang limitation

	NESTED(void, draw, (int x, int y, float c))
	{
		complex float (*out)[X][Y] = p;

		(*out)[x][y] = c * val;
	};

	setup(xiaolin_wu, X, Y, draw, x0, y0, x1, y1);
}

extern void xiaolin_wu_rgba_fl(int X, int Y, float (*out)[X][Y][4], const float (*val)[4], int x0, int y0, int x1, int y1)
{
	void* p = out;	// clang limitation

	NESTED(void, draw, (int x, int y, float c))
	{
		float (*out)[X][Y][4] = p;

		for (int i = 0; i < 4; i++)
			(*out)[x][y][i] = c * (*val)[i];
	};

	setup(xiaolin_wu, X, Y, draw, x0, y0, x1, y1);
}

extern void xiaolin_wu_rgba(int X, int Y, unsigned char (*out)[X][Y][4], const unsigned char (*val)[4], int x0, int y0, int x1, int y1)
{
	void* p = out;	// clang limitation

	NESTED(void, draw, (int x, int y, float c))
	{
		unsigned char (*out)[X][Y][4] = p;

		for (int i = 0; i < 4; i++)
			(*out)[x][y][i] = c * (*val)[i];
	};

	setup(xiaolin_wu, X, Y, draw, x0, y0, x1, y1);
}


#if 0
static double csplineX(double t, const double coeff[4])
{
	return coeff[3] * t + coeff[0] * (1. - t);
}
#endif


static void draw_cspline(int X, int Y, pixel_f out, const double coeff[2][4])
{
	int old[2] = { 0 };

	for (double t = 0.; t <= 1.; t += 0.1) {	// FIXME

		int cur[2];

		cur[0] = (int)cspline(t, coeff[0]);
		cur[1] = (int)cspline(t, coeff[1]);

		if (t > 0.)
			setup(bresenham, X, Y, out, old[0], old[1], cur[0], cur[1]);

		old[0] = cur[0];
		old[1] = cur[1];
	}
}


extern void cspline_cmplx(int X, int Y, complex float (*out)[X][Y], complex float val, const double coeff[2][4])
{
	void* p = out;	// clang limitation

	NESTED(void, draw, (int x, int y, float c))
	{
		complex float (*out)[X][Y] = p;

		(*out)[x][y] = c * val;
	};

	draw_cspline(X, Y, draw, coeff);
}



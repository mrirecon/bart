
extern void bresenham_rgba_fl(int X, int Y, float (*out)[X][Y][4], const float (*val)[4], int x0, int y0, int x1, int y1);
extern void xiaolin_wu_rgba_fl(int X, int Y, float (*out)[X][Y][4], const float (*val)[4], int x0, int y0, int x1, int y1);
extern void bresenham_cmplx(int X, int Y, _Complex float (*out)[X][Y], _Complex float val, int x0, int y0, int x1, int y1);
extern void xiaolin_wu_cmplx(int X, int Y, _Complex float (*out)[X][Y], _Complex float val, int x0, int y0, int x1, int y1);
extern void bresenham_rgba(int X, int Y, unsigned char (*out)[X][Y][4], const unsigned char (*val)[4], int x0, int y0, int x1, int y1);
extern void xiaolin_wu_rgba(int X, int Y, unsigned char (*out)[X][Y][4], const unsigned char (*val)[4], int x0, int y0, int x1, int y1);

extern void cspline_cmplx(int X, int Y, _Complex float (*out)[X][Y], _Complex float val, const double coeff[2][4]);


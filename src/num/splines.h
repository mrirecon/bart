
extern double bernstein(int n, int v, double x);
extern double bezier_curve(double u, int N, const double k[static N + 1]);
extern void bezier_split(double t, int N, double coeffA[static N + 1], double coeffB[static N + 1], const double coeff[static N + 1]);
extern void bezier_increase_degree(int N, double coeff2[static N + 2], const double coeff[static N + 1]);

extern double cspline(double t, const double coeff[4]);

extern double bezier_surface(double u, double v, int N, int M, const double k[static N + 1][M + 1]);
extern double bezier_patch(double u, double v, const double k[4][4]);


extern double bspline(int n, int i, int p, const double tau[static n + 1], double x);
extern double bspline_derivative(int n, int i, int p, const double tau[static n + 1], double x);
extern double bspline_curve(int n, int p, const double t[static n + 1], const double v[static n - p], double x);
extern double bspline_curve_derivative(int k, int n, int p, const double t[static n + 1], const double v[static n - p], double x);
extern void bspline_coeff_derivative_n(int k, int n, int p, double t2[static n - 2 * k + 1], double v2[n - p - k], const double t[static n + 1], const double v[static n - p]);
extern double bspline_curve_zero(int n, int p, const double tau[static n + 1], const double v[static n - p]);

extern void bspline_knot_insert(double x, int n, int p, double t2[static n + 2], double v2[n - p + 1], const double tau[static n + 1], const double v[static n - p]);

extern double nurbs(int n, int p, const double tau[static n + 1], const double coord[static n - p],
	const double w[static n - p], double x);


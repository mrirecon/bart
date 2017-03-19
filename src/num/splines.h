
extern double bernstein(unsigned int n, unsigned int v, double x);
extern double bezier_curve(double u, unsigned int N, const double k[static N + 1]);
extern void bezier_split(double t, unsigned int N, double coeffA[static N + 1], double coeffB[static N + 1], const double coeff[static N + 1]);
extern void bezier_increase_degree(unsigned int N, double coeff2[static N + 2], const double coeff[static N + 1]);

extern double cspline(double t, const double coeff[4]);

extern double bezier_surface(double u, double v, unsigned int N, unsigned int M, const double k[static N + 1][M + 1]);
extern double bezier_patch(double u, double v, const double k[4][4]);


extern double bspline(unsigned int n, unsigned int i, unsigned int p, const double tau[static n + 1], double x);
extern double bspline_derivative(unsigned int n, unsigned int i, unsigned int p, const double tau[static n + 1], double x);
extern double bspline_curve(unsigned int n, unsigned int p, const double t[static n + 1], const double v[static n - p], double x);
extern double bspline_curve_derivative(unsigned int k, unsigned int n, unsigned int p, const double t[static n + 1], const double v[static n - p], double x);
extern void bspline_coeff_derivative_n(unsigned int k, unsigned int n, unsigned int p, double t2[static n - 1], double v2[n - p - 2], const double t[static n + 1], const double v[static n - p]);
extern double bspline_curve_zero(unsigned int n, unsigned int p, const double tau[static n + 1], const double v[static n - p]);

extern void bspline_knot_insert(double x, unsigned int n, unsigned int p, double t2[static n + 2], double v2[n - p + 1], const double tau[static n + 1], const double v[static n - p]);

extern double nurbs(unsigned int n, unsigned int p, const double tau[static n + 1], const double coord[static n - p],
	const double w[static n - p], double x);


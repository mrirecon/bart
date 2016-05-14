
extern double bezier_curve(double u, unsigned int N, const double k[static N + 1]);
extern void bezier_split(double t, unsigned int N, double coeffA[static N + 1], double coeffB[static N + 1], const double coeff[static N + 1]);
extern void bezier_increase_degree(unsigned int N, double coeff2[static N + 2], const double coeff[static N + 1]);

extern double bezier_surface(double u, double v, unsigned int N, unsigned int M, const double k[static N + 1][M + 1]);
extern double bezier_patch(double u, double v, const double k[4][4]);


extern double bspline(unsigned int n, unsigned int i, unsigned int p, const double tau[static n + 1], double x);
extern double bspline_derivative(unsigned int n, unsigned int i, unsigned int p, const double tau[static n + 1], double x);
extern double bspline_curve(unsigned int n, unsigned int p, const double t[static n + 1], const double v[static n + 1], double x);

extern double nurbs(unsigned int n, unsigned int p, const double tau[static n + 1], const double coord[static n + 1 - p],
	const double w[static n + 1 - p], double x);


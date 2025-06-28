
extern int poissondisc(int D, int N, int II, float vardens, float delta, float points[static N][D]);
extern int poissondisc_mc(int D, int T, int N, int II, float vardens,
	const float delta[static T][T], float points[static N][D], int kind[static N]);

extern void mc_poisson_rmatrix(int D, int T, float rmatrix[static T][T], const float delta[static T]);



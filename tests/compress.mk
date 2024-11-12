
tests/test-compress: compress transpose zeros poisson noise fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/zeros 4 32 32 32 4 z.ra					;\
		$(TOOLDIR)/noise z.ra r.ra						;\
		$(TOOLDIR)/poisson -Y32 -Z32 p.ra					;\
		$(TOOLDIR)/transpose 2 13 p.ra p2.ra					;\
		$(TOOLDIR)/transpose 2 13 r.ra r2.ra					;\
		$(TOOLDIR)/fmac p2.ra r2.ra u2.ra					;\
		$(TOOLDIR)/compress u2.ra p2.ra c.ra					;\
		$(TOOLDIR)/compress -d c.ra p2.ra u3.ra					;\
		$(TOOLDIR)/nrmse -t 0. u2.ra u3.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-compress


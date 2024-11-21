
tests/test-estscaling: phantom estscaling pics ones fft fmac nrmse resize
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 64 64 col.ra								;\
	$(TOOLDIR)/phantom -x64 -k ksp.ra							;\
	$(TOOLDIR)/resize -c 0 48 1 48 ksp.ra cal.ra						;\
	$(TOOLDIR)/estscaling -i -x64:64:1 cal.ra scl.ra					;\
	$(TOOLDIR)/fft -i -u 3 ksp.ra fft.ra							;\
	$(TOOLDIR)/fmac fft.ra scl.ra fftscl.ra							;\
	$(TOOLDIR)/pics ksp.ra col.ra recscl.ra							;\
	$(TOOLDIR)/nrmse -t 1.e-6 recscl.ra fftscl.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-estscaling

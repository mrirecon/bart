
tests/test-homodyne: fft rss ones zeros join fmac homodyne nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/fft -iu 3 $(TESTS_OUT)/shepplogan_coil_ksp.ra c0.ra				;\
	$(TOOLDIR)/rss 8 c0.ra r0.ra								;\
	$(TOOLDIR)/ones 2 128 96 o.ra								;\
	$(TOOLDIR)/zeros 2 128 32 z.ra								;\
	$(TOOLDIR)/join 1 o.ra z.ra oz.ra							;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan_coil_ksp.ra oz.ra k1.ra				;\
	$(TOOLDIR)/homodyne -C 1 .75 k1.ra c1.ra						;\
	$(TOOLDIR)/rss 8 c1.ra r1.ra								;\
	$(TOOLDIR)/nrmse -t 0.02 r0.ra r1.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-homodyne-fftu: ones zeros join fmac homodyne rss fft fftmod nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 128 96 o.ra								;\
	$(TOOLDIR)/zeros 2 128 32 z.ra								;\
	$(TOOLDIR)/join 1 o.ra z.ra oz.ra							;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan_coil_ksp.ra oz.ra k1.ra				;\
	$(TOOLDIR)/homodyne -C 1 .75 k1.ra c1.ra						;\
	$(TOOLDIR)/rss 8 c1.ra r1.ra								;\
	$(TOOLDIR)/fft -iu 3 k1.ra c1.ra							;\
	$(TOOLDIR)/fftmod 3 c1.ra c1f.ra							;\
	$(TOOLDIR)/homodyne -n -I -C 1 .75 c1f.ra c1h.ra					;\
	$(TOOLDIR)/rss 8 c1h.ra r2.ra								;\
	$(TOOLDIR)/nrmse -t 0.000001 r1.ra r2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-homodyne tests/test-homodyne-fftu

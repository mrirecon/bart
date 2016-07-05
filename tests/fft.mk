

# basic 2D FFT

$(TESTS_OUT)/shepplogan_fft.ra: fft $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/fft 7 $(TESTS_OUT)/shepplogan.ra $@


tests/test-fft-basic: scale fft nrmse $(TESTS_OUT)/shepplogan_fft.ra $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/scale 16384 $(TESTS_OUT)/shepplogan.ra shepploganS.ra		;\
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_fft.ra shepplogan2.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 shepploganS.ra shepplogan2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	

# unitary FFT

$(TESTS_OUT)/shepplogan_fftu.ra: fft $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/fft -u 7 $(TESTS_OUT)/shepplogan.ra $@

tests/test-fft-unitary: fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/fft -u -i 7 $(TESTS_OUT)/shepplogan_fftu.ra shepplogan2u.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra shepplogan2u.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-fft-basic tests/test-fft-unitary

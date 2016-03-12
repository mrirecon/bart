


# create phantom

$(TESTS_OUT)/shepplogan.ra: phantom
	$(TOOLDIR)/phantom $@


# basic 2D FFT

$(TESTS_OUT)/shepplogan_fft.ra: fft $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/fft 7 $(TESTS_OUT)/shepplogan.ra $@


tests/test-fft-basic: scale fft nrmse $(TESTS_OUT)/shepplogan_fft.ra $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/scale 16384 $(TESTS_OUT)/shepplogan.ra $(TESTS_TMP)/shepploganS.ra
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_fft.ra $(TESTS_TMP)/shepplogan2.ra
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_TMP)/shepploganS.ra $(TESTS_TMP)/shepplogan2.ra
	rm $(TESTS_TMP)/*.ra
	touch $@
	

# unitary FFT

$(TESTS_OUT)/shepplogan_fftu.ra: fft $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/fft -u 7 $(TESTS_OUT)/shepplogan.ra $@

tests/test-fft-unitary: fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	$(TOOLDIR)/fft -u -i 7 $(TESTS_OUT)/shepplogan_fftu.ra $(TESTS_TMP)/shepplogan2u.ra
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra $(TESTS_TMP)/shepplogan2u.ra
	rm $(TESTS_TMP)/*.ra
	touch $@


TESTS += tests/test-fft-basic tests/test-fft-unitary

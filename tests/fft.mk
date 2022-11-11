

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

# uncentered FFT
tests/test-fft-uncentered: fftmod fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/fftmod -i 7 $(TESTS_OUT)/shepplogan_fftu.ra shepplogan_fftu2.ra		;\
	$(TOOLDIR)/fft -uni 7 shepplogan_fftu2.ra shepplogan2u.ra			;\
	$(TOOLDIR)/fftmod -i 7 shepplogan2u.ra shepplogan3u.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra shepplogan3u.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-fft-shift: resize fft fftshift nrmse $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/resize -c 0 127 1 16 3 5 $(TESTS_OUT)/shepplogan_fftu.ra i.ra	;\
	$(TOOLDIR)/fft 15 i.ra k1.ra			;\
	$(TOOLDIR)/fftshift -b 15 i.ra t1.ra		;\
	$(TOOLDIR)/fft -n 15 t1.ra t2.ra						;\
	$(TOOLDIR)/fftshift 15 t2.ra k2.ra					;\
	$(TOOLDIR)/nrmse -t 1e-6 k1.ra k2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-fft-basic tests/test-fft-unitary tests/test-fft-uncentered tests/test-fft-shift

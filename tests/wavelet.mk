
tests/test-wavelet: wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/wavelet 3 $(TESTS_OUT)/shepplogan.ra w.ra				;\
	$(TOOLDIR)/wavelet -a 3 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra a.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-wavelet-haar: wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/wavelet -H 3 $(TESTS_OUT)/shepplogan.ra w.ra				;\
	$(TOOLDIR)/wavelet -H -a 3 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra a.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-wavelet-cdf44: wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/wavelet -C 3 $(TESTS_OUT)/shepplogan.ra w.ra				;\
	$(TOOLDIR)/wavelet -C -a 3 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra a.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-wavelet-batch: ones resize circshift fft fmac wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 1 one.ra							;\
	$(TOOLDIR)/resize -c 2 100 one.ra one2.ra					;\
	$(TOOLDIR)/circshift 2 1 one2.ra one3.ra					;\
	$(TOOLDIR)/fft 4 one3.ra ph.ra							;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan.ra ph.ra ph2.ra				;\
	$(TOOLDIR)/wavelet 3 ph2.ra w.ra						;\
	$(TOOLDIR)/wavelet -a 3 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ph2.ra a.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-wavelet-batch1: ones resize circshift fft fmac wavelet slice nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 1 one.ra							;\
	$(TOOLDIR)/resize -c 2 100 one.ra one2.ra					;\
	$(TOOLDIR)/circshift 2 1 one2.ra one3.ra					;\
	$(TOOLDIR)/fft 4 one3.ra ph.ra							;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan.ra ph.ra ph2.ra				;\
	$(TOOLDIR)/wavelet 3 ph2.ra w.ra						;\
	$(TOOLDIR)/slice 2 80 w.ra w1.ra						;\
	$(TOOLDIR)/slice 2 80 ph2.ra ph1.ra						;\
	$(TOOLDIR)/wavelet 3 ph1.ra a.ra						;\
	$(TOOLDIR)/nrmse -t 0. w1.ra a.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-wavelet-batch2: ones resize circshift fft fmac transpose wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 1 one.ra							;\
	$(TOOLDIR)/resize -c 2 100 one.ra one2.ra					;\
	$(TOOLDIR)/circshift 2 1 one2.ra one3.ra					;\
	$(TOOLDIR)/fft 4 one3.ra ph.ra							;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan.ra ph.ra ph2.ra				;\
	$(TOOLDIR)/transpose 0 2 ph2.ra ph3.ra						;\
	$(TOOLDIR)/wavelet 6 ph3.ra w.ra						;\
	$(TOOLDIR)/wavelet -a 6 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ph3.ra a.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-wavelet-batch3: ones resize circshift fft fmac transpose wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 1 one.ra							;\
	$(TOOLDIR)/resize -c 2 100 one.ra one2.ra					;\
	$(TOOLDIR)/circshift 2 1 one2.ra one3.ra					;\
	$(TOOLDIR)/fft 4 one3.ra ph.ra							;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan.ra ph.ra ph2.ra				;\
	$(TOOLDIR)/transpose 1 2 ph2.ra ph3.ra						;\
	$(TOOLDIR)/wavelet 5 ph3.ra w.ra						;\
	$(TOOLDIR)/wavelet -a 5 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ph3.ra a.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-wavelet-batch4: ones resize circshift fft fmac transpose wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 1 one.ra							;\
	$(TOOLDIR)/resize -c 2 5 3 5 4 5 one.ra one2.ra					;\
	$(TOOLDIR)/circshift 2 1 one2.ra one3.ra					;\
	$(TOOLDIR)/fft 28 one3.ra ph.ra							;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan.ra ph.ra ph2.ra				;\
	$(TOOLDIR)/transpose 0 3 ph2.ra ph3.ra						;\
	$(TOOLDIR)/wavelet 10 ph3.ra w.ra						;\
	$(TOOLDIR)/wavelet -a 10 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ph3.ra a.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-wavelet tests/test-wavelet-batch tests/test-wavelet-batch1
TESTS += tests/test-wavelet-batch2 tests/test-wavelet-batch3 tests/test-wavelet-batch4
TESTS += tests/test-wavelet-haar tests/test-wavelet-cdf44


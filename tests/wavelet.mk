
tests/test-wavelet: wavelet nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/wavelet 3 $(TESTS_OUT)/shepplogan.ra w.ra				;\
	$(TOOLDIR)/wavelet -a 3 128 128 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra a.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-wavelet


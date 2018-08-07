
tests/test-cdf97: cdf97 nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cdf97 3 $(TESTS_OUT)/shepplogan.ra w.ra				;\
	$(TOOLDIR)/cdf97 -i 3 w.ra a.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra a.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-cdf97


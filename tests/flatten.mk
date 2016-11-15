
tests/test-flatten: ones reshape flatten noise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 6 1 2 1 3 1 4 a0.ra						;\
	$(TOOLDIR)/noise a0.ra a1.ra							;\
	$(TOOLDIR)/flatten a1.ra a2.ra							;\
	$(TOOLDIR)/reshape 63 24 1 1 1 1 1 a1.ra a3.ra					;\
	$(TOOLDIR)/nrmse -t 0. a2.ra a3.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-flatten


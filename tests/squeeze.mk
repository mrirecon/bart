
tests/test-squeeze: ones squeeze noise nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 6 1 2 1 3 1 4 a0.ra						;\
	$(TOOLDIR)/ones 3 2 3 4 b0.ra							;\
	$(TOOLDIR)/noise a0.ra a1.ra							;\
	$(TOOLDIR)/noise b0.ra b1.ra							;\
	$(TOOLDIR)/squeeze a1.ra a2.ra							;\
	$(TOOLDIR)/nrmse -t 0. b1.ra a2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-squeeze


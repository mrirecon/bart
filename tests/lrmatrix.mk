


tests/test-lrmatrix: lrmatrix nrmse ones
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 2 10 10 o.ra							;\
	$(TOOLDIR)/lrmatrix -s -o y.ra o.ra x.ra					;\
	$(TOOLDIR)/nrmse -s -t 0.002 o.ra y.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-lrmatrix



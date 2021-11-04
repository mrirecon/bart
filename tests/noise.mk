


tests/test-noise: zeros noise whiten std ones nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 2 100 100 z.ra							;\
	$(TOOLDIR)/noise -s1 -n1. z.ra n.ra						;\
	$(TOOLDIR)/std 3 n.ra d.ra							;\
	$(TOOLDIR)/ones 2 1 1 o.ra							;\
	$(TOOLDIR)/nrmse -t 0.02 o.ra d.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-noise-real: zeros noise whiten std ones nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 2 100 100 z.ra							;\
	$(TOOLDIR)/noise -s1 -n1. -r z.ra n.ra						;\
	$(TOOLDIR)/std 3 n.ra d.ra							;\
	$(TOOLDIR)/ones 2 1 1 o.ra							;\
	$(TOOLDIR)/nrmse -t 0.02 o.ra d.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-noise tests/test-noise-real


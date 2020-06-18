
tests/test-reshape: phantom reshape repmat slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom x.ra								;\
	$(TOOLDIR)/reshape 7 256 1 64 x.ra x2.ra					;\
	$(TOOLDIR)/repmat 1 5 x2.ra x3.ra						;\
	$(TOOLDIR)/reshape 5 128 128 x3.ra x4.ra					;\
	$(TOOLDIR)/slice 1 2 x4.ra x5.ra						;\
	$(TOOLDIR)/reshape 7 128 128 1 x5.ra x6.ra					;\
	$(TOOLDIR)/nrmse -t 0. x.ra x6.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-reshape


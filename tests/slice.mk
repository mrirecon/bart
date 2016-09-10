
tests/test-slice: ones slice resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 1 1 o.ra								;\
	$(TOOLDIR)/resize -c 0 100 1 100 o.ra o0.ra						;\
	$(TOOLDIR)/slice 1 50 o0.ra o1.ra							;\
	$(TOOLDIR)/slice 0 50 o1.ra o2.ra							;\
	$(TOOLDIR)/nrmse -t 0. o2.ra o.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-slice


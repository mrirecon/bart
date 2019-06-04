
tests/test-extract: ones extract resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 10 10 o.ra								;\
	$(TOOLDIR)/resize -c 0 100 1 100 o.ra o0.ra						;\
	$(TOOLDIR)/extract 1 45 55 o0.ra o1.ra							;\
	$(TOOLDIR)/extract 0 45 55 o1.ra o2.ra							;\
	$(TOOLDIR)/nrmse -t 0. o2.ra o.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-extract-multidim: ones extract resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 3 10 10 10 o.ra								;\
	$(TOOLDIR)/resize -c 0 100 1 100 2 100 o.ra o0.ra						;\
	$(TOOLDIR)/extract 0 45 55 1 45 55 2 45 55 o0.ra o1.ra							;\
	$(TOOLDIR)/nrmse -t 0. o1.ra o.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-extract tests/test-extract-multidim


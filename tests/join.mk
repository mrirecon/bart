
tests/test-join: ones join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	seq 1 300 | xargs -P 100 -n1 -I {} $(TOOLDIR)/ones 3 6 1 7 o-{}.ra			;\
	$(TOOLDIR)/ones 3 6 300 7 o1.ra								;\
	$(TOOLDIR)/join 1 o-*.ra o.ra								;\
	$(TOOLDIR)/nrmse -t 0.00001 o.ra o1.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-join-append: ones zeros join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 3 6 7 1 o								;\
	$(TOOLDIR)/zeros 3 6 7 1 z								;\
	$(TOOLDIR)/join 2 o z o j								;\
	$(TOOLDIR)/join -a 2 z o o								;\
	$(TOOLDIR)/nrmse -t 0.00001 o j								;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-join tests/test-join-append


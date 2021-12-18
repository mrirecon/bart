
tests/test-join: ones join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	seq 1 300 | xargs -P 100 -I {} $(TOOLDIR)/ones 3 6 1 7 o-{}.ra				;\
	$(TOOLDIR)/ones 3 6 300 7 o1.ra								;\
	$(TOOLDIR)/join 1 `seq -f "o-%.0f.ra" 1 300` o.ra					;\
	$(TOOLDIR)/nrmse -t 0. o.ra o1.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-join-append: ones zeros join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 3 6 7 1 o								;\
	$(TOOLDIR)/zeros 3 6 7 1 z								;\
	$(TOOLDIR)/join 2 o z o j								;\
	$(TOOLDIR)/join -a 2 z o o								;\
	$(TOOLDIR)/nrmse -t 0. j o								;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-join-append-one: ones zeros join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 3 6 7 1 o								;\
	$(TOOLDIR)/zeros 3 6 7 1 z								;\
	$(TOOLDIR)/join 2 o z j									;\
	$(TOOLDIR)/join -a 2 o z x								;\
	$(TOOLDIR)/nrmse -t 0. j x								;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-join tests/test-join-append tests/test-join-append-one


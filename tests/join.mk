
tests/test-join: ones join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	seq 1 300 | xargs -P 100 -n1 -I {} $(TOOLDIR)/ones 3 6 1 7 o-{}.ra			;\
	$(TOOLDIR)/ones 3 6 300 7 o1.ra								;\
	$(TOOLDIR)/join 1 o-*.ra o.ra								;\
	$(TOOLDIR)/nrmse -t 0.00001 o.ra o1.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-join


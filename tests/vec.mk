


tests/test-vec: ones vec nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 1 2 o.ra								;\
	$(TOOLDIR)/vec 1. 1.0+0.0i v.ra								;\
	$(TOOLDIR)/nrmse -t 0. o.ra v.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-vec-imag: ones scale vec nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 1 2 o1.ra								;\
	$(TOOLDIR)/scale 1.i o1.ra o.ra								;\
	$(TOOLDIR)/vec 1.i 0.0+1.0i v.ra								;\
	$(TOOLDIR)/nrmse -t 0. o.ra v.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-vec tests/test-vec-imag


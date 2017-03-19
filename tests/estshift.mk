
tests/test-estshift: estshift ones resize flip
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 1 1 o.ra								;\
	$(TOOLDIR)/resize -c 0 100 o.ra o0.ra							;\
	$(TOOLDIR)/flip 1 o0.ra o1.ra								;\
	$(TOOLDIR)/estshift 1 o1.ra o1.ra | grep "0.000000"					;\
	$(TOOLDIR)/estshift 1 o0.ra o1.ra | grep "1.000000" 					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-estshift


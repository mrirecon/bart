


tests/test-copy: ones copy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 50 50 a.ra								;\
	$(TOOLDIR)/copy a.ra b.ra								;\
	$(TOOLDIR)/nrmse -t 0. a.ra b.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-copy-out: ones zeros copy resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 50 50 o.ra								;\
	$(TOOLDIR)/resize -c 0 100 1 100 o.ra o0.ra						;\
	$(TOOLDIR)/zeros 2 50 50 o1								;\
	$(TOOLDIR)/copy 0 25 1 25 o0.ra o1							;\
	$(TOOLDIR)/nrmse -t 0. o1 o.ra								;\
	rm *.ra *.hdr *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-copy-in: ones zeros copy resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 50 50 o.ra								;\
	$(TOOLDIR)/resize -c 0 100 1 100 o.ra o0.ra						;\
	$(TOOLDIR)/zeros 2 100 100 o1								;\
	$(TOOLDIR)/copy 0 25 1 25 o.ra o1							;\
	$(TOOLDIR)/nrmse -t 0. o1 o0.ra								;\
	rm *.ra *.hdr *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-copy-in2: ones zeros copy resize nrmse saxpy
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/ones 2 100 100 o0								;\
	$(TOOLDIR)/zeros 2 50 50 z.ra								;\
	$(TOOLDIR)/copy 0 25 1 25 z.ra o0							;\
	$(TOOLDIR)/ones 2 50 50 o.ra								;\
	$(TOOLDIR)/resize -c 0 100 1 100 o.ra i.ra						;\
	$(TOOLDIR)/ones 2 100 100 o.ra								;\
	$(TOOLDIR)/saxpy -- -1 i.ra o.ra o1.ra							;\
	$(TOOLDIR)/nrmse -t 0. o1.ra o0								;\
	rm *.ra *.hdr *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-copy tests/test-copy-out tests/test-copy-in tests/test-copy-in2


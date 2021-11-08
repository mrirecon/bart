
tests/test-multicfl: multicfl ones nrmse $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/ones 7 3 5 1 1 2 1 5 o.ra									;\
	$(TOOLDIR)/multicfl $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra o.ra comb	;\
	$(TOOLDIR)/multicfl -s comb sc sck o2									;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_coil.ra sc						;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_coil_ksp.ra sck						;\
	$(TOOLDIR)/nrmse -t 0. o.ra o2										;\
	rm *.ra ; rm *.cfl; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-multicfl


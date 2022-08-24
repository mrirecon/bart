

# test python readcfl and writecfl interface


tests/test-python-cfl: $(TOOLDIR)/tests/pythoncfl.py nrmse flip $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/flip 0 $(TESTS_OUT)/shepplogan.ra shepplogan					;\
	PYTHONPATH=$(TOOLDIR)/python $(TOOLDIR)/tests/pythoncfl.py shepplogan shepplogan2	;\
	$(TOOLDIR)/nrmse -t 0.000001 shepplogan shepplogan2					;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-python-multicfl: $(TOOLDIR)/tests/pythonmulticfl.py multicfl ones nrmse $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/ones 7 3 5 1 1 2 1 5 o.ra									;\
	$(TOOLDIR)/multicfl $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra o.ra comb	;\
	PYTHONPATH=$(TOOLDIR)/python $(TOOLDIR)/tests/pythonmulticfl.py comb comb2				;\
	$(TOOLDIR)/multicfl -s comb2 sc sck o2									;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_coil.ra sc						;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_coil_ksp.ra sck						;\
	$(TOOLDIR)/nrmse -t 0. o.ra o2										;\
	rm *.ra ; rm *.cfl; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS_PYTHON += tests/test-python-cfl tests/test-python-multicfl




# test python reacfl and writecfl interface


tests/test-python-cfl: $(TOOLDIR)/tests/pythoncfl.py nrmse flip $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/flip 0 $(TESTS_OUT)/shepplogan.ra shepplogan					;\
	PYTHONPATH=$(TOOLDIR)/python $(TOOLDIR)/tests/pythoncfl.py shepplogan shepplogan2	;\
	$(TOOLDIR)/nrmse -t 0.000001 shepplogan shepplogan2					;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
	

TESTS_PYTHON += tests/test-python-cfl


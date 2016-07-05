

tests/test-nrmse-scale: nrmse scale $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/scale 2.i $(TESTS_OUT)/shepplogan.ra shepplogan_sc.ra		;\
	$(TOOLDIR)/nrmse -s -t 0.000001 $(TESTS_OUT)/shepplogan.ra shepplogan_sc.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-nrmse-scale

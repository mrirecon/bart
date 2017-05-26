


tests/test-nlinv: bart nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra			;\
	$(TOOLDIR)/bart normalize 8 c.ra c_norm.ra					;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-nlinv

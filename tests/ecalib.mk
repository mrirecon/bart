


tests/test-ecalib: ecalib pocsense nrmse $(TESTS_OUT)/shepplogan_coil.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ecalib -m1 $(TESTS_OUT)/shepplogan_coil.ra coils.ra			;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil.ra coils.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-ecalib


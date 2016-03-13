


tests/test-ecalib: ecalib pocsense nrmse $(TESTS_OUT)/shepplogan_coil.ra
	$(TOOLDIR)/ecalib -m1 $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_TMP)/coils.ra
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_TMP)/coils.ra $(TESTS_TMP)/proj.ra
	$(TOOLDIR)/nrmse -t 0.05 $(TESTS_TMP)/proj.ra $(TESTS_OUT)/shepplogan_coil.ra
	rm $(TESTS_TMP)/*.ra
	touch $@


TESTS += tests/test-ecalib


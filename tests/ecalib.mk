


tests/test-ecalib: ecalib pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ecalib -m1 $(TESTS_OUT)/shepplogan_coil_ksp.ra coils.ra		;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra coils.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ecalib-auto: ecalib pocsense nrmse noise $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) ;\
	$(TOOLDIR)/noise -n 100 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_noise.ra ;\
	$(TOOLDIR)/ecalib -m 1 -a -v 100 shepplogan_noise.ra coils.ra ;\
	$(TOOLDIR)/pocsense -i 1 shepplogan_noise.ra coils.ra proj.ra ;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-ecalib tests/test-ecalib-auto

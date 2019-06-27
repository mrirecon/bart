


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
	$(TOOLDIR)/nrmse -t 0.035 $(TESTS_OUT)/shepplogan_coil_ksp.ra proj.ra;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ecalib-rotation: ecalib cc nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/ecalib -m1 ksp-cc.ra scc-sens.ra				;\
	$(TOOLDIR)/ecalib -P -m1 ksp-cc.ra scc-sens2.ra					;\
	$(TOOLDIR)/nrmse -s -t 0.000001 scc-sens.ra scc-sens2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ecalib-rotation2: ecalib cc fmac transpose nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ecalib -m1 $(TESTS_OUT)/shepplogan_coil_ksp.ra sens.ra		;\
	$(TOOLDIR)/cc -M -S $(TESTS_OUT)/shepplogan_coil_ksp.ra cc-mat.ra		;\
	$(TOOLDIR)/fmac -C -s 8 $(TESTS_OUT)/shepplogan_coil_ksp.ra cc-mat.ra ksp-cc.ra	;\
	$(TOOLDIR)/transpose 3 4 ksp-cc.ra ksp-cc-2.ra					;\
	$(TOOLDIR)/ecalib -P -m1 ksp-cc-2.ra scc-sens-1.ra				;\
	$(TOOLDIR)/transpose 3 4 scc-sens-1.ra scc-sens-2.ra				;\
	$(TOOLDIR)/fmac -s 16 scc-sens-2.ra cc-mat.ra scc-sens.ra			;\
	$(TOOLDIR)/nrmse -t 0.1 sens.ra scc-sens.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-ecalib tests/test-ecalib-auto tests/test-ecalib-rotation
TESTS += tests/test-ecalib-rotation2




tests/test-nlinv: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra			;\
	$(TOOLDIR)/normalize 8 c.ra c_norm.ra						;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-batch: conj join nlinv fmac fft nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/conj $(TESTS_OUT)/shepplogan_coil_ksp.ra kc.ra			;\
	$(TOOLDIR)/join 6 $(TESTS_OUT)/shepplogan_coil_ksp.ra kc.ra ksp.ra		;\
	$(TOOLDIR)/nlinv -N -S ksp.ra r.ra c.ra						;\
	$(TOOLDIR)/fmac r.ra c.ra x.ra							;\
	$(TOOLDIR)/fft -u 7 x.ra ksp2.ra						;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra ksp2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-batch2: repmat nlinv fmac fft nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 7 2 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra		;\
	$(TOOLDIR)/nlinv -s128 -N -S ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/fmac r.ra c.ra x.ra							;\
	$(TOOLDIR)/fft -u 7 x.ra ksp2.ra						;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra ksp2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-nlinv tests/test-nlinv-batch tests/test-nlinv-batch2


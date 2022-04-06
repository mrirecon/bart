
tests/test-fakeksp-phantom: scale fakeksp nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/coils.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/scale 128. $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp_scaled.ra				;\
	$(TOOLDIR)/fakeksp $(TESTS_OUT)/shepplogan.ra ksp_scaled.ra $(TESTS_OUT)/coils.ra out.ra	;\
	$(TOOLDIR)/nrmse -t 0.22 ksp_scaled.ra out.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-fakeksp-phantom-replace: scale upat squeeze fmac fakeksp nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/coils.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/scale 128. $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp_scaled.ra				;\
	$(TOOLDIR)/upat -z2 u.ra									;\
	$(TOOLDIR)/squeeze u.ra u2.ra									;\
	$(TOOLDIR)/fmac ksp_scaled.ra u2.ra ksp_under.ra						;\
	$(TOOLDIR)/fakeksp -r $(TESTS_OUT)/shepplogan.ra ksp_under.ra $(TESTS_OUT)/coils.ra out.ra	;\
	$(TOOLDIR)/fmac out.ra u2.ra out_pat.ra								;\
	$(TOOLDIR)/nrmse -t 0.0 ksp_under.ra out_pat.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-fakeksp-phantom tests/test-fakeksp-phantom-replace


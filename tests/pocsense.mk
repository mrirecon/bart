
# pocsense with -r0.0 actually calls pocsense without the prox-operator used for regularization
tests/test-pocsense-pi: pocsense normalize noise nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                                    ;\
	$(TOOLDIR)/normalize 8 $(TESTS_OUT)/coils.ra coilsnorm.ra			;\
	$(TOOLDIR)/noise -n 10 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_noise.ra ;\
	$(TOOLDIR)/pocsense -r0.0 shepplogan_noise.ra coilsnorm.ra reco.ra   		;\
	$(TOOLDIR)/nrmse -t 0.05 reco.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pocsense-pi-reg: pocsense normalize noise nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                                    ;\
	$(TOOLDIR)/normalize 8 $(TESTS_OUT)/coils.ra coilsnorm.ra			;\
	$(TOOLDIR)/noise -n 10 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_noise.ra ;\
	$(TOOLDIR)/pocsense -r0.01 shepplogan_noise.ra coilsnorm.ra reco.ra   		;\
	$(TOOLDIR)/nrmse -t 0.05 reco.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pocsense-pi-wl: pocsense normalize noise nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                                    ;\
	$(TOOLDIR)/normalize 8 $(TESTS_OUT)/coils.ra coilsnorm.ra			;\
	$(TOOLDIR)/noise -n 10 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_noise.ra ;\
	$(TOOLDIR)/pocsense -l1 -r0.01 shepplogan_noise.ra coilsnorm.ra reco.ra    	;\
	$(TOOLDIR)/nrmse -t 0.05 reco.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra    	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-pocsense-pi tests/test-pocsense-pi-reg tests/test-pocsense-pi-wl

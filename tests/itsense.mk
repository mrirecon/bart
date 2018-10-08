


tests/test-itsense: itsense scale pattern nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/pattern  $(TESTS_OUT)/shepplogan_coil_ksp.ra pat.ra						;\
	$(TOOLDIR)/itsense 0.001 $(TESTS_OUT)/coils.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra pat.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra									;\
	$(TOOLDIR)/nrmse -t 0.23 reco2.ra $(TESTS_OUT)/shepplogan.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-itsense

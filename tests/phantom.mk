
$(TESTS_OUT)/shepplogan.ra: phantom
	$(TOOLDIR)/phantom $@

$(TESTS_OUT)/shepplogan_ksp.ra: phantom
	$(TOOLDIR)/phantom -k $@


tests/test-phantom-ksp: fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_ksp.ra
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_ksp.ra $(TESTS_TMP)/shepplogan_img.ra
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_OUT)/shepplogan.ra $(TESTS_TMP)/shepplogan_img.ra
	rm $(TESTS_TMP)/*.ra
	touch $@

tests/test-phantom-noncart: traj phantom reshape nrmse $(TESTS_OUT)/shepplogan_ksp.ra
	$(TOOLDIR)/traj $(TESTS_TMP)/traj
	$(TOOLDIR)/phantom -k -t $(TESTS_TMP)/traj $(TESTS_TMP)/shepplogan_ksp2.ra
	$(TOOLDIR)/reshape 7 128 128 1 $(TESTS_TMP)/shepplogan_ksp2.ra $(TESTS_TMP)/shepplogan_ksp3.ra
	$(TOOLDIR)/nrmse -t 0.00001 $(TESTS_OUT)/shepplogan_ksp.ra $(TESTS_TMP)/shepplogan_ksp3.ra
	rm $(TESTS_TMP)/*.ra
	touch $@


TESTS += tests/test-phantom-ksp tests/test-phantom-noncart


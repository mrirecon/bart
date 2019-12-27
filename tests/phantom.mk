
$(TESTS_OUT)/shepplogan.ra: phantom
	$(TOOLDIR)/phantom $@

$(TESTS_OUT)/shepplogan_ksp.ra: phantom
	$(TOOLDIR)/phantom -k $@

$(TESTS_OUT)/shepplogan_coil.ra: phantom
	$(TOOLDIR)/phantom -s8 $@

$(TESTS_OUT)/shepplogan_coil_ksp.ra: phantom
	$(TOOLDIR)/phantom -s8 -k $@

$(TESTS_OUT)/coils.ra: phantom
	$(TOOLDIR)/phantom -S8 $@


tests/test-phantom-ksp: fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_ksp.ra shepplogan_img.ra			;\
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_OUT)/shepplogan.ra shepplogan_img.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-noncart: traj phantom reshape nrmse $(TESTS_OUT)/shepplogan_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj traj.ra									;\
	$(TOOLDIR)/phantom -k -t traj.ra shepplogan_ksp2.ra					;\
	$(TOOLDIR)/reshape 7 128 128 1 shepplogan_ksp2.ra shepplogan_ksp3.ra			;\
	$(TOOLDIR)/nrmse -t 0.00001 $(TESTS_OUT)/shepplogan_ksp.ra shepplogan_ksp3.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP) 
	touch $@


tests/test-phantom-coil: fmac nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/coils.ra sl_coil2.ra		;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_coil.ra sl_coil2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-ksp-coil: fft nrmse $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_cimg.ra		;\
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_OUT)/shepplogan_coil.ra shepplogan_cimg.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-bart: fft nrmse phantom
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -B -k k.ra								;\
	$(TOOLDIR)/fft -i 3 k.ra x.ra								;\
	$(TOOLDIR)/phantom -B r.ra								;\
	$(TOOLDIR)/nrmse -t 0.21 r.ra x.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-phantom-basis: nrmse phantom fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phantom -T -k k0.ra								;\
	$(TOOLDIR)/phantom -T -b -k k1.ra							;\
	$(TOOLDIR)/fmac -s 64 k1.ra k2.ra							;\
	$(TOOLDIR)/nrmse -t 0.000001 k0.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-phantom-ksp tests/test-phantom-noncart tests/test-phantom-coil tests/test-phantom-ksp-coil
TESTS += tests/test-phantom-bart tests/test-phantom-basis


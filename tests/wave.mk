
tests/test-wave: wavepsf fft resize fmac wave nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                          ;\
	$(TOOLDIR)/wavepsf -x 640 -y 128 wave_psf.ra                          ;\
	$(TOOLDIR)/fft -iu 7 $(TESTS_OUT)/shepplogan_coil_ksp.ra img.ra       ;\
	$(TOOLDIR)/resize -c 0 640 img.ra wave_zpad.ra                        ;\
	$(TOOLDIR)/fft -u 1 wave_zpad.ra wave_hyb.ra                          ;\
	$(TOOLDIR)/fmac wave_hyb.ra wave_psf.ra wave_acq.ra                   ;\
	$(TOOLDIR)/fft -u 6 wave_acq.ra wave_ksp.ra                           ;\
	$(TOOLDIR)/wave $(TESTS_OUT)/coils.ra wave_psf.ra wave_ksp.ra reco.ra ;\
	$(TOOLDIR)/nrmse -t 0.23 -s reco.ra $(TESTS_OUT)/shepplogan.ra        ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-wave

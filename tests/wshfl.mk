tests/test-wshfl: wavepsf fmac fft resize transpose squeeze poly join wshfl nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                          ;\
	$(TOOLDIR)/wavepsf -x 640 -y 128 wave_psf.ra                          ;\
	$(TOOLDIR)/fft -iu 7 $(TESTS_OUT)/shepplogan_coil_ksp.ra img.ra       ;\
	$(TOOLDIR)/resize -c 0 640 img.ra wave_zpad.ra                        ;\
	$(TOOLDIR)/fft -u 1 wave_zpad.ra wave_hyb.ra                          ;\
	$(TOOLDIR)/fmac wave_hyb.ra wave_psf.ra wave_acq.ra                   ;\
	$(TOOLDIR)/fft -u 6 wave_acq.ra wave_ksp.ra                           ;\
	$(TOOLDIR)/transpose 1 3 wave_ksp.ra permuted.ra                      ;\
	$(TOOLDIR)/squeeze permuted.ra table.ra                               ;\
	$(TOOLDIR)/poly 128 1 0 1 ky.ra                                       ;\
	$(TOOLDIR)/poly 128 1 0 0 kz.ra                                       ;\
	$(TOOLDIR)/poly 128 1 0 0 te.ra                                       ;\
	$(TOOLDIR)/join 1 ky.ra kz.ra te.ra reorder.ra                        ;\
	$(TOOLDIR)/poly 1 0 1 phi.ra                                          ;\
	$(TOOLDIR)/wshfl $(TESTS_OUT)/coils.ra wave_psf.ra phi.ra reorder.ra table.ra reco.ra ;\
	$(TOOLDIR)/nrmse -t 0.23 -s $(TESTS_OUT)/shepplogan.ra reco.ra        ;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-wshfl

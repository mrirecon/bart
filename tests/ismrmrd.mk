# Test BART with ISMRMRD.

tests/test-ismrmrd-read: $(ISMRM_BASE)/build/utilities/ismrmrd_generate_cartesian_shepp_logan $(TESTS_DIR)/ismrmrd_h5tocfl.py ismrmrd nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(ISMRM_BASE)/build/utilities/ismrmrd_generate_cartesian_shepp_logan -m128 -c1 -o mrd_phantom.h5;\
	$(TOOLDIR)/ismrmrd mrd_phantom.h5 mrd_phantom							;\
	python3 $(TESTS_DIR)/ismrmrd_h5tocfl.py mrd_phantom.h5 mrd_phantom_py				;\
	$(TOOLDIR)/nrmse -t0 mrd_phantom_py mrd_phantom							;\
	rm *.cfl *.hdr *.h5; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS_ISMRMRD += tests/test-ismrmrd-read

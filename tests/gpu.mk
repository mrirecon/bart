

# 

tests/test-pics-gpu: phantom pics nrmse $(TESTS_OUT)/shepplogan_coil.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -S8 coils.ra							;\
	$(TOOLDIR)/pics -g -S -r0.001 $(TESTS_OUT)/shepplogan_coil.ra coils.ra reco1.ra	;\
	$(TOOLDIR)/pics    -S -r0.001 $(TESTS_OUT)/shepplogan_coil.ra coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_GPU += tests/test-pics-gpu


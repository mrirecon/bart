


tests/test-cc-svd: cc resize nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -S -p 8 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc8.ra		;\
	$(TOOLDIR)/cc -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc4.ra		;\
	$(TOOLDIR)/resize 3 8 ksp-cc4.ra ksp-cc-z.ra					;\
	$(TOOLDIR)/nrmse -t 0.1 ksp-cc8.ra ksp-cc-z.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-cc-geom: cc rss resize nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -G -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc4.ra		;\
	$(TOOLDIR)/rss 11 ksp-cc4.ra ksp0-cc4.ra					;\
	$(TOOLDIR)/rss 11 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp0.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 ksp0.ra ksp0-cc4.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-cc-esp: cc rss resize nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -E -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc4.ra		;\
	$(TOOLDIR)/resize 3 8 ksp-cc4.ra ksp-cc-z.ra					;\
	$(TOOLDIR)/rss 11 ksp-cc4.ra ksp0-cc4.ra					;\
	$(TOOLDIR)/rss 11 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp0.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 ksp0.ra ksp0-cc4.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-cc-svd-matrix: cc extract fmac transpose nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/cc -M -S $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/fmac -C -s 8 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat-4.ra ksp-cc-3.ra	;\
	$(TOOLDIR)/transpose 3 4 ksp-cc-3.ra ksp-cc-4.ra				;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-4.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-cc-svd tests/test-cc-geom tests/test-cc-esp


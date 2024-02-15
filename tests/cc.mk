


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
	$(TOOLDIR)/extract 4 0 4 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/fmac -C -s 8 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat-4.ra ksp-cc-3.ra	;\
	$(TOOLDIR)/transpose 3 4 ksp-cc-3.ra ksp-cc-4.ra				;\
	$(TOOLDIR)/nrmse -t 0. ksp-cc.ra ksp-cc-4.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-cc-rovir: bart $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) ; export BART_TOOLBOX_DIR=$(ROOTDIR)	;\
	$(ROOTDIR)/bart ones 2 16 8 o							;\
	$(ROOTDIR)/bart resize 1 16 o pos						;\
	$(ROOTDIR)/bart flip 2 pos neg							;\
	$(ROOTDIR)/scripts/rovir.sh -p4 $(TESTS_OUT)/shepplogan_coil_ksp.ra pos neg ksp1;\
	$(ROOTDIR)/scripts/rovir.sh -p4 $(TESTS_OUT)/shepplogan_coil_ksp.ra neg pos ksp2;\
	$(ROOTDIR)/bart nlinv -S ksp1 img1						;\
	$(ROOTDIR)/bart nlinv -S ksp2 img2						;\
	$(ROOTDIR)/bart resize 1 64 img1 opt						;\
	$(ROOTDIR)/bart resize 1 64 img2 sup						;\
	$(ROOTDIR)/bart fmac -C -s 3 opt opt energy_opt2				;\
	$(ROOTDIR)/bart fmac -C -s 3 sup sup energy_sup					;\
	$(ROOTDIR)/bart creal energy_opt2 energy_opt					;\
	$(ROOTDIR)/bart scale 50 energy_sup energy_sup_scl				;\
	$(ROOTDIR)/bart join 0 energy_sup_scl energy_opt energy_joined			;\
	$(ROOTDIR)/bart mip 1 energy_joined energy_max					;\
	$(ROOTDIR)/bart nrmse -t 0. energy_opt energy_max				;\
	rm *.{cfl,hdr} ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-cc-rovir-noncart: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) ; export BART_TOOLBOX_DIR=$(ROOTDIR)	;\
	$(ROOTDIR)/bart traj -r trj							;\
	$(ROOTDIR)/bart phantom -s8 -k -ttrj ksp					;\
	$(ROOTDIR)/bart ones 2 16 8 o							;\
	$(ROOTDIR)/bart resize 1 16 o pos						;\
	$(ROOTDIR)/bart flip 2 pos neg							;\
	$(ROOTDIR)/scripts/rovir.sh -p4 -t trj ksp pos neg ksp1				;\
	$(ROOTDIR)/scripts/rovir.sh -p4 -t trj ksp neg pos ksp2				;\
	$(ROOTDIR)/bart nlinv -S -t trj ksp1 img1					;\
	$(ROOTDIR)/bart nlinv -S -t trj ksp2 img2					;\
	$(ROOTDIR)/bart resize 1 64 img1 opt						;\
	$(ROOTDIR)/bart resize 1 64 img2 sup						;\
	$(ROOTDIR)/bart fmac -C -s 3 opt opt energy_opt2				;\
	$(ROOTDIR)/bart fmac -C -s 3 sup sup energy_sup					;\
	$(ROOTDIR)/bart creal energy_opt2 energy_opt					;\
	$(ROOTDIR)/bart scale 100 energy_sup energy_sup_scl				;\
	$(ROOTDIR)/bart join 0 energy_sup_scl energy_opt energy_joined			;\
	$(ROOTDIR)/bart mip 1 energy_joined energy_max					;\
	$(ROOTDIR)/bart nrmse -t 0. energy_opt energy_max				;\
	rm *.{cfl,hdr} ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-cc-svd tests/test-cc-geom tests/test-cc-esp tests/test-cc-svd-matrix
TESTS += tests/test-cc-rovir tests/test-cc-rovir-noncart


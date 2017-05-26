


tests/test-ccapply-forward: cc ccapply nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/cc -M -S $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc-2.ra	;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-backward: cc ccapply extract fmac transpose nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -M -S $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc.ra	;\
	$(TOOLDIR)/ccapply -S -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/transpose 3 4 ksp-cc.ra ksp-ccT.ra					;\
	$(TOOLDIR)/fmac -s 16 ksp-ccT.ra sccmat-4.ra ksp-3.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 ksp-2.ra ksp-3.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-geom-forward: cc ccapply extract nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -G -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/cc -M -p 4 -G $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -G -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc-2.ra	;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-geom-backward: cc ccapply nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -M -p4 -G $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -G -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc.ra	;\
	$(TOOLDIR)/ccapply -G -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-2.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-esp-forward: cc ccapply extract nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -E -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/cc -M -p 4 -E $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -E -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc-2.ra	;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-esp-backward: cc ccapply nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -M -p4 -E $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -E -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc.ra	;\
	$(TOOLDIR)/ccapply -E -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-2.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@







TESTS += tests/test-ccapply-forward tests/test-ccapply-backward
TESTS += tests/test-ccapply-geom-forward tests/test-ccapply-geom-backward
TESTS += tests/test-ccapply-esp-forward tests/test-ccapply-esp-backward





tests/test-ccapply-forward: phantom cc ccapply nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -S -p 4 ksp.ra ksp-cc.ra						;\
	$(TOOLDIR)/cc -M -S ksp.ra sccmat.ra						;\
	$(TOOLDIR)/ccapply -S -p 4 ksp.ra sccmat.ra ksp-cc-2.ra				;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-backward: phantom cc ccapply extract fmac transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -M -S ksp.ra sccmat.ra						;\
	$(TOOLDIR)/ccapply -S -p 4 ksp.ra sccmat.ra ksp-cc.ra				;\
	$(TOOLDIR)/ccapply -S -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/transpose 3 4 ksp-cc.ra ksp-ccT.ra					;\
	$(TOOLDIR)/fmac -s 16 ksp-ccT.ra sccmat-4.ra ksp-3.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 ksp.ra ksp-2.ra					;\
	$(TOOLDIR)/nrmse -t 0.0001 ksp-2.ra ksp-3.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-geom-forward: phantom cc ccapply extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -G -p 4 ksp.ra ksp-cc.ra						;\
	$(TOOLDIR)/cc -M -p 4 -G ksp.ra sccmat.ra					;\
	$(TOOLDIR)/ccapply -G -p 4 ksp.ra sccmat.ra ksp-cc-2.ra				;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-geom-backward: phantom cc ccapply nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -M -p4 -G ksp.ra sccmat.ra					;\
	$(TOOLDIR)/ccapply -G -p 4 ksp.ra sccmat.ra ksp-cc.ra				;\
	$(TOOLDIR)/ccapply -G -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 ksp.ra ksp-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-esp-forward: phantom cc ccapply extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -E -p 4 ksp.ra ksp-cc.ra						;\
	$(TOOLDIR)/cc -M -p 4 -E ksp.ra sccmat.ra					;\
	$(TOOLDIR)/ccapply -E -p 4 ksp.ra sccmat.ra ksp-cc-2.ra				;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-esp-backward: phantom cc ccapply nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -M -p4 -E ksp.ra sccmat.ra					;\
	$(TOOLDIR)/ccapply -E -p 4 ksp.ra sccmat.ra ksp-cc.ra				;\
	$(TOOLDIR)/ccapply -E -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 ksp.ra ksp-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@







TESTS += tests/test-ccapply-forward tests/test-ccapply-backward
TESTS += tests/test-ccapply-geom-forward tests/test-ccapply-geom-backward
TESTS += tests/test-ccapply-esp-forward tests/test-ccapply-esp-backward


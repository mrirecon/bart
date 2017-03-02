


tests/test-ccapply-forward: phantom cc ccapply extract fmac transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -S -p 4 ksp.ra ksp-cc.ra						;\
	$(TOOLDIR)/cc -S ksp.ra sccmat.ra						;\
	$(TOOLDIR)/ccapply -S -p 4 ksp.ra sccmat.ra ksp-cc-2.ra				;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/fmac -C -s 8 ksp.ra sccmat-4.ra ksp-cc-3.ra				;\
	$(TOOLDIR)/transpose 3 4 ksp-cc-3.ra ksp-cc-3.ra				;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-2.ra					;\
	$(TOOLDIR)/nrmse -t 0.001 ksp-cc.ra ksp-cc-3.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ccapply-backward: phantom cc ccapply extract fmac transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s 8 -k ksp.ra						;\
	$(TOOLDIR)/cc -S ksp.ra sccmat.ra						;\
	$(TOOLDIR)/ccapply -S -p 4 ksp.ra sccmat.ra ksp-cc.ra				;\
	$(TOOLDIR)/ccapply -S -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/extract 4 0 3 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/transpose 3 4 ksp-cc.ra ksp-cc.ra					;\
	$(TOOLDIR)/fmac -s 16 ksp-cc.ra sccmat-4.ra ksp-3.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 ksp.ra ksp-2.ra					;\
	$(TOOLDIR)/nrmse -t 0.0001 ksp-2.ra ksp-3.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



TESTS += tests/test-ccapply-forward tests/test-ccapply-backward


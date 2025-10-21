


tests/test-ccapply-forward: cc ccapply nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/cc -M -S $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc-2.ra	;\
	$(TOOLDIR)/nrmse -t 0. ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-backward: cc ccapply extract fmac transpose nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -M -S $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -S -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc.ra	;\
	$(TOOLDIR)/ccapply -S -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/extract 4 0 4 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/transpose 3 4 ksp-cc.ra ksp-ccT.ra					;\
	$(TOOLDIR)/fmac -s 16 ksp-ccT.ra sccmat-4.ra ksp-3.ra				;\
	$(TOOLDIR)/nrmse -t 0.08 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-2.ra		;\
	$(TOOLDIR)/nrmse -t 0. ksp-2.ra ksp-3.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-geom-forward: cc ccapply extract nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -G -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/cc -M -p 4 -G $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -G -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc-2.ra	;\
	$(TOOLDIR)/extract 4 0 4 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/nrmse -t 0. ksp-cc.ra ksp-cc-2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-geom-backward: cc ccapply nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -M -p4 -G $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -G -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc.ra	;\
	$(TOOLDIR)/ccapply -G -u ksp-cc.ra sccmat.ra ksp-2.ra				;\
	$(TOOLDIR)/nrmse -t 0.008 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-2.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ccapply-esp-forward: cc ccapply extract nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/cc -E -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp-cc.ra		;\
	$(TOOLDIR)/cc -M -p 4 -E $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra		;\
	$(TOOLDIR)/ccapply -E -p 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra sccmat.ra ksp-cc-2.ra	;\
	$(TOOLDIR)/extract 4 0 4 sccmat.ra sccmat-4.ra					;\
	$(TOOLDIR)/nrmse -t 0. ksp-cc.ra ksp-cc-2.ra					;\
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


tests/test-ccapply-rgc-forward: bart cc ccapply copy nrmse fft transpose traj phantom extract join repmat reshape bitmask
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -r -o2 -D -c -x128 -y19 -t5 -- -					|\
	$(TOOLDIR)/phantom -k -s8 -t - phantom_full.ra						;\
	$(TOOLDIR)/extract 3 0 4 phantom_full.ra phantom_1.ra					;\
	$(TOOLDIR)/extract 3 4 8 phantom_full.ra phantom_2.ra					;\
	$(TOOLDIR)/repmat -- 4 2 phantom_1.ra -							|\
	$(TOOLDIR)/reshape -- $$($(TOOLDIR)/bitmask 3 4) 8 1 - phantom_1.ra			;\
	$(TOOLDIR)/repmat -- 4 2 phantom_2.ra -							|\
	$(TOOLDIR)/reshape -- $$($(TOOLDIR)/bitmask 3 4) 8 1 - phantom_2.ra			;\
	$(TOOLDIR)/join -- 11 phantom_1.ra phantom_2.ra -					|\
	$(TOOLDIR)/reshape -- $$($(TOOLDIR)/bitmask 10 11) 10 1 - phantom.ra			;\
	$(TOOLDIR)/copy --stream 1024 -- phantom.ra -						|\
	$(ROOTDIR)/bart -r - cc -A -M -- - $(TESTS_OUT)/ccmat.ra				;\
	$(TOOLDIR)/ccapply -p4 -A10 phantom.ra $(TESTS_OUT)/ccmat.ra $(TESTS_OUT)/ksp-cc.ra	;\
	$(TOOLDIR)/transpose -- 10 0 phantom.ra -						|\
	$(TOOLDIR)/fft -u -- 1  - -								|\
	$(TOOLDIR)/cc -A -M -G -- - $(TESTS_OUT)/ccmat_G.ra					;\
	$(TOOLDIR)/transpose -- 10 0 phantom.ra -						|\
	$(TOOLDIR)/ccapply -p4 -t -G -- - $(TESTS_OUT)/ccmat_G.ra -				|\
	$(TOOLDIR)/transpose -- 0 10 - $(TESTS_OUT)/ksp-cc_G.ra					;\
	$(TOOLDIR)/nrmse -t 0.0001 $(TESTS_OUT)/ksp-cc_G.ra $(TESTS_OUT)/ksp-cc.ra		;\
	rm *.ra ; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@



TESTS += tests/test-ccapply-forward tests/test-ccapply-backward
TESTS += tests/test-ccapply-geom-forward tests/test-ccapply-geom-backward
TESTS += tests/test-ccapply-esp-forward tests/test-ccapply-esp-backward
TESTS += tests/test-ccapply-rgc-forward


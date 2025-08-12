
tests/test-seq-raga: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x 512 -y 377 -r -A -s 1 --double-base trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 -t377 --raga grad.ra mom.ra samples.ra 		;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga


tests/test-seq-raga-sms: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x 512 -y 377 -m3 -r -A -s 1 --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 -t377 --raga --mb_factor=3 -m3 grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra					;\
	$(TOOLDIR)/nrmse -t 3e-7 trj_ref.ra trj_seq.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms


tests/test-seq-raga-sms-al: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 512 -y 377 -m3 -r -A -s 1 -l --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 -t377 --raga_al --mb_factor=3 -m3 grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al


tests/test-seq-raga-sms-al-frame: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 512 -y 377 -m3 -r -A -s 1 -l -t3 --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 -t1131 --raga_al --mb_factor=3 -m3 grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al-frame


tests/test-seq-offcenter: seq traj extract scale phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/traj -x 512 -y 377 -r -A -s 1 --double-base trj_ref.ra				;\
	$(TOOLDIR)/seq -s 25.6:12.8:0 -r 377 -t377 --raga --no-spoiling grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra							;\
	$(TOOLDIR)/extract 0 1 2 samples.ra adc_phase.ra						;\
	$(TOOLDIR)/scale 0.5 trj_ref.ra trj_scale.ra							;\
	$(TOOLDIR)/phantom -k -t trj_scale.ra ksp.ra							;\
	$(TOOLDIR)/fovshift -s 0.1:0.05:0 -t trj_ref.ra ksp.ra ksp_ref.ra				;\
	$(TOOLDIR)/fmac adc_phase.ra ksp.ra ksp_seq.ra							;\
	$(TOOLDIR)/nrmse -t 1e-6 ksp_ref.ra ksp_seq.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-offcenter


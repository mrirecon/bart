
tests/test-seq-raga: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -D trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --raga grad.ra mom.ra samples.ra 	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra		;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga

tests/test-seq-raga-chrono: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -A -s 1 --double-base trj_ref.ra ;\
	$(TOOLDIR)/seq -r 377 --raga --chrono grad.ra mom.ra samples.ra 	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-chrono

tests/test-seq-raga-sms: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 --raga --chrono --mb_factor=3 -m3 grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 3e-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms


tests/test-seq-raga-sms-al: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 -l --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 --raga_al --chrono --mb_factor=3 -m3 grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al


tests/test-seq-raga-sms-al-frame: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 -l -t3 --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 -t1131 --raga_al --chrono --mb_factor=3 -m3 grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al-frame


tests/test-seq-offcenter: seq traj extract scale phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -A -s 1 --double-base trj_ref.ra				;\
	$(TOOLDIR)/seq -s 25.6:12.8:0 -r 377 --raga --chrono --no-spoiling grad.ra mom.ra samples.ra	;\
	$(TOOLDIR)/extract 0 2 5 samples.ra trj_seq.ra							;\
	$(TOOLDIR)/extract 0 1 2 samples.ra adc_phase.ra						;\
	$(TOOLDIR)/scale 0.5 trj_ref.ra trj_scale.ra							;\
	$(TOOLDIR)/phantom -k -t trj_scale.ra ksp.ra							;\
	$(TOOLDIR)/fovshift -s 0.2:0.1:0 -t trj_ref.ra ksp.ra ksp_ref.ra				;\
	$(TOOLDIR)/fmac adc_phase.ra ksp.ra ksp_seq.ra							;\
	$(TOOLDIR)/nrmse -t 1e-6 ksp_ref.ra ksp_seq.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-offcenter



tests/test-seq-raga-ordering: seq traj extract raga bin nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/seq -r 377 --raga grad.ra mom.ra samples.ra 				;\
	$(TOOLDIR)/seq -r 377 --raga --chrono grad_c.ra mom_c.ra samples_c.ra 	;\
	$(TOOLDIR)/raga 377 ind.ra								;\
	$(TOOLDIR)/bin -o ind.ra grad_c.ra grad_c_sort.ra					;\
	$(TOOLDIR)/nrmse -t 0 grad.ra grad_c_sort.ra						;\
	$(TOOLDIR)/bin -o ind.ra mom_c.ra mom_c_sort.ra						;\
	$(TOOLDIR)/nrmse -t 0 mom.ra mom_c_sort.ra						;\
	$(TOOLDIR)/bin -o ind.ra samples_c.ra samples_c_sort.ra					;\
	$(TOOLDIR)/nrmse -t 0 samples.ra samples_c_sort.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-ordering
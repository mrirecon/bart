
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


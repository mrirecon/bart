
tests/test-seq-raga: seq traj transpose slice join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x 512 -y 377 -r -A -s 1 --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 -t377 --pe_mode 0 grad.ra mom.ra samples.ra 		;\
	$(TOOLDIR)/transpose 1 2 samples.ra   samples_t.ra				;\
	$(TOOLDIR)/transpose 0 1 samples_t.ra samples_tt.ra				;\
	$(TOOLDIR)/slice 4 3 samples_tt.ra samples_t0.ra 				;\
	$(TOOLDIR)/slice 4 2 samples_tt.ra samples_t1.ra 				;\
	$(TOOLDIR)/slice 4 4 samples_tt.ra samples_t2.ra 				;\
	$(TOOLDIR)/join 0 samples_t0.ra samples_t1.ra samples_t2.ra trj_seq.ra		;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga


tests/test-seq-raga-sms: seq traj transpose slice join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/traj -x 512 -y 377 -m3 -r -A -s 1 --double-base trj_ref.ra 					;\
	$(TOOLDIR)/seq -r 377 -t377 --pe_mode 0 --mb_factor=3 -m3 grad.ra mom.ra samples.ra			;\
	$(TOOLDIR)/transpose 1 2 samples.ra   samples_t.ra							;\
	$(TOOLDIR)/transpose 0 1 samples_t.ra samples_tt.ra							;\
	$(TOOLDIR)/slice 4 3 samples_tt.ra samples_t0.ra 							;\
	$(TOOLDIR)/slice 4 2 samples_tt.ra samples_t1.ra 							;\
	$(TOOLDIR)/slice 4 4 samples_tt.ra samples_t2.ra 							;\
	$(TOOLDIR)/join 0 samples_t0.ra samples_t1.ra samples_t2.ra trj_seq.ra 					;\
	$(TOOLDIR)/nrmse -t 3e-7 trj_ref.ra trj_seq.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms


tests/test-seq-raga-sms-al: seq traj transpose slice join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/traj -x 512 -y 377 -m3 -r -A -s 1 -l --double-base trj_ref.ra 				;\
	$(TOOLDIR)/seq -r 377 -t377 --pe_mode 1 --mb_factor=3 -m3 grad.ra mom.ra samples.ra			;\
	$(TOOLDIR)/transpose 1 2 samples.ra   samples_t.ra							;\
	$(TOOLDIR)/transpose 0 1 samples_t.ra samples_tt.ra							;\
	$(TOOLDIR)/slice 4 3 samples_tt.ra samples_t0.ra 							;\
	$(TOOLDIR)/slice 4 2 samples_tt.ra samples_t1.ra 							;\
	$(TOOLDIR)/slice 4 4 samples_tt.ra samples_t2.ra 							;\
	$(TOOLDIR)/join 0 samples_t0.ra samples_t1.ra samples_t2.ra trj_seq.ra 					;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al

tests/test-seq-raga-sms-al-frame: seq traj transpose slice join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/traj -x 512 -y 377 -m3 -r -A -s 1 -l -t3 --double-base trj_ref.ra 				;\
	$(TOOLDIR)/seq -r 377 -t1131 --pe_mode 1 --mb_factor=3 -m3 grad.ra mom.ra samples.ra			;\
	$(TOOLDIR)/transpose 1 2 samples.ra   samples_t.ra							;\
	$(TOOLDIR)/transpose 0 1 samples_t.ra samples_tt.ra							;\
	$(TOOLDIR)/slice 4 3 samples_tt.ra samples_t0.ra 							;\
	$(TOOLDIR)/slice 4 2 samples_tt.ra samples_t1.ra 							;\
	$(TOOLDIR)/slice 4 4 samples_tt.ra samples_t2.ra 							;\
	$(TOOLDIR)/join 0 samples_t0.ra samples_t1.ra samples_t2.ra trj_seq.ra 					;\
	$(TOOLDIR)/nrmse -t 2e-7 trj_ref.ra trj_seq.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al-frame


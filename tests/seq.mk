
tests/test-seq-raga: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -D trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --raga samples.ra grad.ra mom.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra		;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga

tests/test-seq-raga2: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -D trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --dwell 4.2E-6 --raga samples.ra grad.ra mom.ra 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra		;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga2

tests/test-seq-raga-chrono: seq traj extract nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -A -s 1 --double-base trj_ref.ra ;\
	$(TOOLDIR)/seq -r 377 --raga --chrono samples.ra		 	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra				;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-chrono

tests/test-seq-raga-sms: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 --double-base trj_ref.ra 	;\
	$(TOOLDIR)/seq -r 377 --raga --chrono --mb_factor=3 -m3 samples.ra		;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra					;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms


tests/test-seq-raga-sms-al: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 -l --double-base trj_ref.ra 		;\
	$(TOOLDIR)/seq -r 377 --raga --raga_flags 8192 --chrono --mb_factor=3 -m3 samples.ra	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al


tests/test-seq-raga-sms-al-frame: seq traj extract nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -m3 -r -A -s 1 -l -t3 --double-base trj_ref.ra	;\
	$(TOOLDIR)/seq -r 377 -t1131 --raga --raga_flags 8192 --chrono --mb_factor=3 -m3 samples.ra		;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra						;\
	$(TOOLDIR)/nrmse -t 3E-7 trj_ref.ra trj_seq.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-sms-al-frame


tests/test-seq-offcenter: seq traj extract scale phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x 256 -o 2. -y 377 -r -A -s 1 --double-base trj_ref.ra		;\
	$(TOOLDIR)/seq -s 0.0256:0.0128:0 -r 377 --raga --chrono --no-spoiling samples.ra	;\
	$(TOOLDIR)/extract 0 0 3 samples.ra trj_seq.ra					;\
	$(TOOLDIR)/extract 0 4 5 samples.ra adc_phase.ra				;\
	$(TOOLDIR)/scale 0.5 trj_ref.ra trj_scale.ra					;\
	$(TOOLDIR)/phantom -k -t trj_scale.ra ksp.ra					;\
	$(TOOLDIR)/fovshift -s 0.1:0.05:0 -t trj_ref.ra ksp.ra ksp_ref.ra		;\
	$(TOOLDIR)/fmac adc_phase.ra ksp.ra ksp_seq.ra					;\
	$(TOOLDIR)/nrmse -t 1E-6 ksp_ref.ra ksp_seq.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-offcenter


tests/test-seq-relative-fovshift: seq traj extract scale phantom fovshift fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/seq -s 0.0256:0.0128:0	-r 377 --no-spoiling samples_abs.ra	;\
	$(TOOLDIR)/seq -S 0.1:0.05:0 		-r 377 --no-spoiling samples_rel.ra	;\
	$(TOOLDIR)/nrmse -t 0 samples_abs.ra samples_rel.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-relative-fovshift


tests/test-seq-raga-ordering: seq traj extract raga bin nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/seq -r 377 --raga samples.ra grad.ra mom.ra 			;\
	$(TOOLDIR)/seq -r 377 --raga --chrono samples_c.ra grad_c.ra mom_c.ra	;\
	$(TOOLDIR)/raga 377 ind.ra						;\
	$(TOOLDIR)/bin -o ind.ra grad_c.ra grad_c_sort.ra			;\
	$(TOOLDIR)/nrmse -t 0 grad.ra grad_c_sort.ra				;\
	$(TOOLDIR)/bin -o ind.ra mom_c.ra mom_c_sort.ra				;\
	$(TOOLDIR)/nrmse -t 0 mom.ra mom_c_sort.ra				;\
	$(TOOLDIR)/bin -o ind.ra samples_c.ra samples_c_sort.ra			;\
	$(TOOLDIR)/nrmse -t 0 samples.ra samples_c_sort.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-ordering


tests/test-seq-raga-ind: seq raga nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/raga -s1 377 ind_ref.ra		;\
	$(TOOLDIR)/seq -r 377 --raga -R ind_seq.ra  	;\
	$(TOOLDIR)/nrmse -t 0. ind_ref.ra ind_seq.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-ind


tests/test-seq-raga-ind-multislice: seq raga nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/raga -s1 -m3 377 ind_ref.ra		;\
	$(TOOLDIR)/seq -r 377 -m3 --raga -R ind_seq.ra  ;\
	$(TOOLDIR)/nrmse -t 0. ind_ref.ra ind_seq.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-raga-ind-multislice

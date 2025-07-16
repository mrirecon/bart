
tests/test-seq-traj: seq traj transpose slice join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -r -D -s1 -x512 -y1 -t200 trj_ref.ra					;\
	$(TOOLDIR)/seq -t200 --pe_mode 2 --norm-kspace grad.ra mom.ra samples.ra ;\
	$(TOOLDIR)/transpose 0 1 samples.ra samples_t.ra				;\
	$(TOOLDIR)/slice 4 3 samples_t.ra samples_t0.ra 				;\
	$(TOOLDIR)/slice 4 2 samples_t.ra samples_t1.ra 				;\
	$(TOOLDIR)/slice 4 4 samples_t.ra samples_t2.ra 				;\
	$(TOOLDIR)/join 0 samples_t0.ra samples_t1.ra samples_t2.ra trj_seq.ra ;\
	$(TOOLDIR)/nrmse -t 0.000006 trj_ref.ra trj_seq.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS +=  tests/test-seq-traj

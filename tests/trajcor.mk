
tests/test-trajcor: traj trajcor nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -y10  -s1 traj.ra				;\
	$(TOOLDIR)/traj -r -y10  -s1 -O -q1:2:3 traj_gd.ra		;\
	$(TOOLDIR)/trajcor -q1:2:3 traj.ra traj_cor.ra			;\
	$(TOOLDIR)/nrmse -t1.e-6 traj_gd.ra traj_cor.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-trajcor2: traj trajcor nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -y10 -t3 -s1 traj.ra				;\
	$(TOOLDIR)/traj -r -y10 -t3 -s1 -O -q-1:0.5:1 traj_gd.ra	;\
	$(TOOLDIR)/trajcor -q-1:0.5:1 traj.ra traj_cor.ra		;\
	$(TOOLDIR)/nrmse -t1.e-6 traj_gd.ra traj_cor.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-trajcor tests/test-trajcor2

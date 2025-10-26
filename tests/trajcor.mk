TOL=1e-7

tests/test-trajcor: traj trajcor nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -y10 -s1 traj.ra				;\
	$(TOOLDIR)/traj -r -y10 -s1 -O -q1:2:3 traj_gd.ra		;\
	$(TOOLDIR)/trajcor -q1:2:3 -O traj.ra traj_cor.ra		;\
	$(TOOLDIR)/nrmse -t$(TOL) traj_gd.ra traj_cor.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-trajcor2: traj trajcor nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -y10 -t3 -s1 traj.ra				;\
	$(TOOLDIR)/traj -r -y10 -t3 -s1 -O -q-1:0.5:1 traj_gd.ra	;\
	$(TOOLDIR)/trajcor -q-1:0.5:1 -O traj.ra traj_cor.ra		;\
	$(TOOLDIR)/nrmse -t$(TOL) traj_gd.ra traj_cor.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-trajcor3: traj trajcor nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -y10 -s1 traj.ra				;\
	$(TOOLDIR)/traj -r -y10 -s1 -q1:2:3 traj_gd.ra			;\
	$(TOOLDIR)/trajcor -q1:2:3 traj.ra traj_cor.ra			;\
	$(TOOLDIR)/nrmse -t$(TOL) traj_gd.ra traj_cor.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-trajcor4: traj trajcor nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	OPTS="-r -y65 -x128 -o2 -D"					;\
	DELAY="-q1:2:3"							;\
	$(TOOLDIR)/traj $$OPTS traj.ra					;\
	$(TOOLDIR)/traj $$OPTS $$DELAY traj_gd.ra			;\
	$(TOOLDIR)/trajcor $$DELAY traj.ra traj_cor.ra			;\
	$(TOOLDIR)/nrmse -t$(TOL) traj_gd.ra traj_cor.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-trajcor-file: traj trajcor vec nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/vec -- -1 0.5 1 qf.ra				;\
	$(TOOLDIR)/traj -r -y10 -t3 -s1 traj.ra				;\
	$(TOOLDIR)/trajcor -Vqf.ra    traj.ra traj1.ra			;\
	$(TOOLDIR)/trajcor -q-1:0.5:1 traj.ra traj2.ra			;\
	$(TOOLDIR)/nrmse -t$(TOL) traj1.ra traj2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-trajcor tests/test-trajcor2 tests/test-trajcor3 tests/test-trajcor4 tests/test-trajcor-file

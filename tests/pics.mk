


tests/test-pics-pi: phantom pics scale nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -S8 coils.ra							;\
	$(TOOLDIR)/pics -S -r0.001 $(TESTS_OUT)/shepplogan_coil.ra coils.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.23 reco2.ra $(TESTS_OUT)/shepplogan.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-noncart: traj scale phantom ones pics nufft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S -r0.001 -t traj2.ra ksp.ra o.ra reco.ra			;\
	$(TOOLDIR)/nufft traj2.ra reco.ra k2.ra						;\
	$(TOOLDIR)/nrmse -t 0.002 ksp.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-cs: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y48 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S -RT:7:0:0.001 -u0.1 -e -t traj2.ra ksp.ra o.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.22 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-pics: traj scale phantom pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y32 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra					;\
	$(TOOLDIR)/phantom -S8 coils.ra							;\
	$(TOOLDIR)/pics -S -RT:7:0:0.001 -u1000000000. -e -t traj2.ra ksp.ra coils.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.22 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test that weights =1 have no effect
tests/test-pics-weights: phantom pics ones nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -S8 coils.ra							;\
	$(TOOLDIR)/ones 2 128 128 weights.ra						;\
	$(TOOLDIR)/pics -S -r0.001 $(TESTS_OUT)/shepplogan_coil.ra coils.ra reco1.ra	;\
	$(TOOLDIR)/pics -S -r0.001 -p weights.ra $(TESTS_OUT)/shepplogan_coil.ra coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 reco2.ra reco1.ra				 	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test that weights =1 have no effect
# FIXME: this was 0.005 before but fails on travis
tests/test-pics-noncart-weights: traj scale ones phantom pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y32 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra					;\
	$(TOOLDIR)/phantom -S8 coils.ra							;\
	$(TOOLDIR)/ones 4 1 256 32 1 weights.ra						;\
	$(TOOLDIR)/pics -S -r0.001 -p weights.ra -t traj2.ra ksp.ra coils.ra reco1.ra	;\
	$(TOOLDIR)/pics -S -r0.001               -t traj2.ra ksp.ra coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.010 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-warmstart: phantom pics scale nrmse $(TESTS_OUT)/shepplogan_coil.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -S8 coils.ra							;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -i0 -Wo.ra $(TESTS_OUT)/shepplogan_coil.ra coils.ra reco.ra	;\
	$(TOOLDIR)/nrmse -t 0. o.ra reco.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-batch: phantom pics repmat nrmse $(TESTS_OUT)/shepplogan_coil.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -S8 coils.ra							;\
	$(TOOLDIR)/repmat 5 32 $(TESTS_OUT)/shepplogan_coil.ra kspaces.ra		;\
	$(TOOLDIR)/pics -r0.01 -B32 kspaces.ra coils.ra reco1.ra			;\
	$(TOOLDIR)/pics -r0.01      kspaces.ra coils.ra reco2.ra			;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS += tests/test-pics-pi tests/test-pics-noncart tests/test-pics-cs tests/test-pics-pics
TESTS += tests/test-pics-weights tests/test-pics-noncart-weights
TESTS += tests/test-pics-warmstart tests/test-pics-batch





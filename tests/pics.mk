


tests/test-pics-pi: phantom pics scale nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil.ra
	$(TOOLDIR)/phantom -S8 $(TESTS_TMP)/coils.ra
	$(TOOLDIR)/pics -S -r0.001 $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_TMP)/coils.ra $(TESTS_TMP)/reco.ra
	$(TOOLDIR)/scale 128. $(TESTS_TMP)/reco.ra $(TESTS_TMP)/reco2.ra
	$(TOOLDIR)/nrmse -t 0.23 $(TESTS_TMP)/reco2.ra $(TESTS_OUT)/shepplogan.ra
	rm $(TESTS_TMP)/*.ra
	touch $@



tests/test-pics-noncart: traj scale phantom ones pics nufft nrmse
	$(TOOLDIR)/traj -r -x256 -y64 $(TESTS_TMP)/traj.ra
	$(TOOLDIR)/scale 0.5 $(TESTS_TMP)/traj.ra $(TESTS_TMP)/traj2.ra
	$(TOOLDIR)/phantom -t $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/ksp.ra
	$(TOOLDIR)/ones 3 128 128 1 $(TESTS_TMP)/o.ra
	$(TOOLDIR)/pics -S -r0.001 -t $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/ksp.ra $(TESTS_TMP)/o.ra $(TESTS_TMP)/reco.ra
	$(TOOLDIR)/nufft $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/reco.ra $(TESTS_TMP)/k2.ra
	$(TOOLDIR)/nrmse -t 0.002 $(TESTS_TMP)/ksp.ra $(TESTS_TMP)/k2.ra
	rm $(TESTS_TMP)/*.ra
	touch $@



tests/test-pics-cs: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/traj -r -x256 -y48 $(TESTS_TMP)/traj.ra
	$(TOOLDIR)/scale 0.5 $(TESTS_TMP)/traj.ra $(TESTS_TMP)/traj2.ra
	$(TOOLDIR)/phantom -t $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/ksp.ra
	$(TOOLDIR)/ones 3 128 128 1 $(TESTS_TMP)/o.ra
	$(TOOLDIR)/pics -S -RT:7:0:0.001 -u0.1 -e -t $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/ksp.ra $(TESTS_TMP)/o.ra $(TESTS_TMP)/reco.ra
	$(TOOLDIR)/scale 128. $(TESTS_TMP)/reco.ra $(TESTS_TMP)/reco2.ra
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_TMP)/reco2.ra $(TESTS_OUT)/shepplogan.ra
	rm $(TESTS_TMP)/*.ra
	touch $@



tests/test-pics-pics: traj scale phantom pics nrmse $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/traj -r -x256 -y32 $(TESTS_TMP)/traj.ra
	$(TOOLDIR)/scale 0.5 $(TESTS_TMP)/traj.ra $(TESTS_TMP)/traj2.ra
	$(TOOLDIR)/phantom -s8 -t $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/ksp.ra
	$(TOOLDIR)/phantom -S8 $(TESTS_TMP)/coils.ra
	$(TOOLDIR)/pics -S -RT:7:0:0.001 -u1000000000. -e -t $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/ksp.ra $(TESTS_TMP)/coils.ra $(TESTS_TMP)/reco.ra
	$(TOOLDIR)/scale 128. $(TESTS_TMP)/reco.ra $(TESTS_TMP)/reco2.ra
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_TMP)/reco2.ra $(TESTS_OUT)/shepplogan.ra
	rm $(TESTS_TMP)/*.ra
	touch $@


TESTS += tests/test-pics-pi tests/test-pics-noncart tests/test-pics-cs tests/test-pics-pics





# compare customAngle to default angle

tests/test-traj-custom: traj scale phantom nufft poly nrmse 
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -x128 -y128 -r traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/bart phantom -t traj2.ra -k ksp.ra				;\
	$(TOOLDIR)/nufft -a traj2.ra ksp.ra reco_default.ra			;\
	$(TOOLDIR)/bart poly 128 1 0 0.0245436926 angle.ra			;\
	$(TOOLDIR)/traj -x128 -y128 -r -C angle.ra traj.ra			;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/bart phantom -t traj2.ra -k ksp.ra				;\
	$(TOOLDIR)/nufft -a traj2.ra ksp.ra reco_custom.ra			;\
	$(TOOLDIR)/nrmse -t 0.0001 reco_default.ra reco_custom.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-custom
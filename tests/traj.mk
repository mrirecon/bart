# compare customAngle to default angle


tests/test-traj-custom: traj poly nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y128 -r traja.ra				;\
	$(TOOLDIR)/poly 128 1 0 0.0245436926 angle.ra			;\
	$(TOOLDIR)/traj -x128 -y128 -r -C angle.ra trajb.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 traja.ra trajb.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-custom

tests/test-traj-3D: traj ones scale slice rss nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -3 -x128 -y128 -r traj.ra			;\
	$(TOOLDIR)/ones 3 1 1 128 o.ra					;\
	$(TOOLDIR)/scale 63.5 o.ra a.ra					;\
	$(TOOLDIR)/slice 1 0 traj.ra t.ra				;\
	$(TOOLDIR)/rss 1 t.ra b.ra					;\
	$(TOOLDIR)/nrmse -t 0.0000001 b.ra a.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-3D


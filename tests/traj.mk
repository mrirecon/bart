
tests/test-traj-over: traj scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y71 -r traja.ra				;\
	$(TOOLDIR)/traj -x64 -y71 -o2. -r trajb.ra			;\
	$(TOOLDIR)/scale 0.5 traja.ra traja2.ra				;\
	$(TOOLDIR)/nrmse -t 0.0000001 traja2.ra trajb.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-over


tests/test-traj-dccen: traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y71 -r -c traja.ra			;\
	$(TOOLDIR)/traj -x128 -y71 -q-0.5:-0.5:0. -r trajb.ra		;\
	$(TOOLDIR)/nrmse -t 0.0000001 traja.ra trajb.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-dccen



tests/test-traj-dccen-over: traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x64 -y71 -r -c -o2. traja.ra			;\
	$(TOOLDIR)/traj -x64 -y71 -q-0.5:-0.5:0. -r -o2. trajb.ra	;\
	$(TOOLDIR)/nrmse -t 0.0000001 traja.ra trajb.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-dccen-over



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


tests/test-traj-rot: traj phantom estshift vec nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -R0. -r -y360 -D t0.ra 				;\
	$(TOOLDIR)/phantom -k -t t0.ra k0.ra 				;\
	$(TOOLDIR)/traj -R30. -r -y360 -D t30.ra			;\
	$(TOOLDIR)/phantom -k -t t30.ra k30.ra 				;\
	$(TOOLDIR)/vec 30. real_shift.ra 				;\
	$(TOOLDIR)/vec `$(TOOLDIR)/estshift 4 k0.ra k30.ra | grep -Eo "[0-9.]+$$"` shift.ra		;\
	$(TOOLDIR)/nrmse -t1e-6 real_shift.ra shift.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rot


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


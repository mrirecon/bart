
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


tests/test-traj-rational-approx-loop: traj slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 233 -r -A --double-base -t 2 traj.ra			;\
	$(TOOLDIR)/slice 10 0 traj.ra o1.ra			;\
	$(TOOLDIR)/slice 10 1 traj.ra o2.ra			;\
	$(TOOLDIR)/nrmse -t 0.007 o1.ra o2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-loop


tests/test-traj-rational-approx-pattern: traj ones nufft fft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 233 -r -A --double-base t.ra				;\
	$(TOOLDIR)/traj -y 233 -r -D t2.ra					;\
	$(TOOLDIR)/ones 3 1 128 233 o.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t.ra o.ra psf.ra		;\
	$(TOOLDIR)/fft 7 psf.ra pattern.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t2.ra o.ra psf2.ra		;\
	$(TOOLDIR)/fft 7 psf2.ra pattern2.ra				;\
	$(TOOLDIR)/nrmse -t 0.0005 pattern.ra pattern2.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-pattern


tests/test-traj-rational-approx-pattern2: traj ones nufft fft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 466 -r -A t.ra				;\
	$(TOOLDIR)/traj -y 466 -r -D t2.ra					;\
	$(TOOLDIR)/ones 3 1 128 466 o.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t.ra o.ra psf.ra		;\
	$(TOOLDIR)/fft 7 psf.ra pattern.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t2.ra o.ra psf2.ra		;\
	$(TOOLDIR)/fft 7 psf2.ra pattern2.ra				;\
	$(TOOLDIR)/nrmse -t 0.0005 pattern.ra pattern2.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-pattern2


tests/test-traj-double-base: traj slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 2 -r -G -s1 --double-base traj.ra		;\
	$(TOOLDIR)/traj -y 3 -r -G -s1 traj2.ra				;\
	$(TOOLDIR)/slice 2 1 traj.ra o1.ra				;\
	$(TOOLDIR)/slice 2 2 traj2.ra o2.ra				;\
	$(TOOLDIR)/nrmse -t 0.007 o1.ra o2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-double-base
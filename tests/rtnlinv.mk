

tests/test-rtnlinv: traj scale phantom nufft rtnlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x128 -y21 -t5 traj.ra			;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra				;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra			;\
	$(TOOLDIR)/nufft -a traj.ra ksp.ra I.ra				;\
	$(TOOLDIR)/rtnlinv -N -S -i7 -t traj.ra ksp.ra r.ra c.ra	;\
	$(TOOLDIR)/fmac r.ra c.ra x.ra					;\
	$(TOOLDIR)/nufft traj.ra x.ra k2.ra				;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra k2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-rtnlinv-precomp: traj scale phantom ones repmat fft nufft rtnlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -x128 -y21 -t5 traj.ra				;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra					;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra				;\
	$(TOOLDIR)/ones 3 1 128 21 o.ra						;\
	$(TOOLDIR)/repmat 10 5 o.ra o2.ra					;\
	$(TOOLDIR)/nufft -a traj.ra o2.ra psf.ra				;\
	$(TOOLDIR)/nufft -a traj.ra ksp.ra adj.ra				;\
	$(TOOLDIR)/fft -u 7 psf.ra mtf.ra					;\
	$(TOOLDIR)/fft -u 7 adj.ra ksp2.ra					;\
	$(TOOLDIR)/rtnlinv -w1. -N -i7 -p mtf.ra ksp2.ra r1.ra c1.ra		;\
	$(TOOLDIR)/rtnlinv -w1. -N -i7 -t traj.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/fmac r1.ra c1.ra x1.ra					;\
	$(TOOLDIR)/fmac r2.ra c2.ra x2.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra r1.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 c2.ra c1.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-rtnlinv tests/test-rtnlinv-precomp


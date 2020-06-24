

tests/test-rtnlinv: traj scale phantom nufft resize rtnlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x128 -y21 -t5 traj.ra			;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra				;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra			;\
	$(TOOLDIR)/nufft -a traj.ra ksp.ra I.ra				;\
	$(TOOLDIR)/rtnlinv -N -S -i9 -t traj2.ra ksp.ra r.ra c.ra	;\
	$(TOOLDIR)/resize -c 0 64 1 64 c.ra c2.ra			;\
	$(TOOLDIR)/fmac r.ra c2.ra x.ra					;\
	$(TOOLDIR)/nufft traj2.ra x.ra k2.ra				;\
	$(TOOLDIR)/scale 2. k2.ra k3.ra					;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra k3.ra				;\
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
	$(TOOLDIR)/scale 4. mtf.ra mtf2.ra					;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -p mtf2.ra ksp2.ra r1.ra c1.ra		;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -t traj2.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra r1.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 c2.ra c1.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-rtnlinv-nlinv-noncart: traj scale phantom rtnlinv nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -x128 -y21 traj.ra				;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra					;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra				;\
	$(TOOLDIR)/nlinv -w1. -N -i9 -t traj2.ra ksp.ra r1.ra c1.ra		;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -t traj2.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra r1.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 c2.ra c1.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-rtnlinv-nlinv-pseudocart: scale phantom ones rtnlinv nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -s8 -k -x 128  ksp.ra				;\
	$(TOOLDIR)/ones 2 128 128 psf.ra				;\
	$(TOOLDIR)/nlinv -w1. -N -i9 -f1. -n ksp.ra r1.ra c1.ra		;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -f1. -p psf.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 r2.ra r1.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 c2.ra c1.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-rtnlinv tests/test-rtnlinv-precomp tests/test-rtnlinv-nlinv-noncart tests/test-rtnlinv-nlinv-pseudocart
#TESTS += tests/test-rtnlinv-precomp


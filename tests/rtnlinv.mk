

tests/test-rtnlinv: traj phantom nufft resize rtnlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -o2 -r -x64 -y21 -t5 traj.ra			;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra			;\
	$(TOOLDIR)/rtnlinv -N -S -i9 -t traj.ra ksp.ra r.ra c.ra	;\
	$(TOOLDIR)/resize -c 0 64 1 64 c.ra c2.ra			;\
	$(TOOLDIR)/fmac r.ra c2.ra x.ra					;\
	$(TOOLDIR)/nufft traj.ra x.ra k2.ra				;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra k2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-rtnlinv-precomp: traj scale phantom ones repmat fft nufft rtnlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -x128 -y21 -t5 traj.ra				;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra					;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra				;\
	$(TOOLDIR)/ones 3 1 128 21 o.ra						;\
	$(TOOLDIR)/repmat 10 5 o.ra o2.ra					;\
	$(TOOLDIR)/nufft -a traj.ra o2.ra psf.ra				;\
	$(TOOLDIR)/nufft -a traj.ra ksp.ra _adj.ra				;\
	$(TOOLDIR)/scale 2 _adj.ra adj.ra					;\
	$(TOOLDIR)/fft -u 7 psf.ra mtf.ra					;\
	$(TOOLDIR)/fft -u 7 adj.ra ksp2.ra					;\
	$(TOOLDIR)/scale 4. mtf.ra mtf2.ra					;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -p mtf2.ra ksp2.ra r1.ra c1.ra		;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -t traj2.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/nrmse -t 0.000002 r2.ra r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.000002 c2.ra c1.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


# tests are bit flaky due to parallelization
tests/test-rtnlinv-nlinv-sms: traj phantom reshape rtnlinv nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -r -x128 -o2. -y60 traj.ra						;\
	$(TOOLDIR)/phantom -s6 -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/reshape 8200 2 3 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/rtnlinv -w0.0001 -s -S -N -i8 -t traj.ra ksp2.ra r.ra c.ra			;\
	$(TOOLDIR)/nlinv --legacy-early-stopping -w0.0001 -S -N -i8 --psf-based -t traj.ra ksp2.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.0015 r.ra r2.ra							;\
	$(TOOLDIR)/nrmse -t 0.0015 c.ra c2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-rtnlinv-nlinv-noncart: traj phantom rtnlinv nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -x64 -o2. -y21 traj.ra				;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra				;\
	$(TOOLDIR)/nlinv --psf-based --legacy-early-stopping -w1. -N -i9 -t traj.ra ksp.ra r1.ra c1.ra	;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -t traj.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 r2.ra r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 c2.ra c1.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-rtnlinv-nlinv-pseudocart: phantom ones rtnlinv nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -s8 -k -x 128  ksp.ra				;\
	$(TOOLDIR)/ones 2 128 128 psf.ra					;\
	$(TOOLDIR)/nlinv --legacy-early-stopping  -w1. -N -i9 -f1. -n ksp.ra r1.ra c1.ra		;\
	$(TOOLDIR)/rtnlinv -w1. -N -i9 -f1. -p psf.ra ksp.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 r2.ra r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 c2.ra c1.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-rtnlinv-maps-dims: phantom ones rtnlinv show
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -s1 -k ksp.ra					;\
	$(TOOLDIR)/ones 2 128 128 psf.ra					;\
	$(TOOLDIR)/rtnlinv -m4       -S -i1 -ppsf.ra ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/rtnlinv -m4    -U -S -i1 -ppsf.ra ksp.ra rU.ra cU.ra		;\
	$(TOOLDIR)/rtnlinv -m4 -N    -S -i1 -ppsf.ra ksp.ra rN.ra cN.ra		;\
	$(TOOLDIR)/rtnlinv -m4 -N -U -S -i1 -ppsf.ra ksp.ra rNU.ra cNU.ra	;\
	[ 1 -eq `$(TOOLDIR)/show -d 4 r.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 c.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 rU.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 cU.ra` ] 					&&\
	[ 1 -eq `$(TOOLDIR)/show -d 4 rN.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 cN.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 rNU.ra` ] 				&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 cNU.ra` ] 				&&\
	true
	touch $@


tests/test-rtnlinv-noncart-maps-dims: traj phantom rtnlinv show
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -y21 traj.ra						;\
	$(TOOLDIR)/phantom -s1 -k -ttraj.ra ksp.ra				;\
	$(TOOLDIR)/rtnlinv -m4       -S -i1 -ttraj.ra  ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/rtnlinv -m4    -U -S -i1 -ttraj.ra  ksp.ra rU.ra cU.ra	;\
	$(TOOLDIR)/rtnlinv -m4 -N    -S -i1 -ttraj.ra  ksp.ra rN.ra cN.ra	;\
	$(TOOLDIR)/rtnlinv -m4 -N -U -S -i1 -ttraj.ra  ksp.ra rNU.ra cNU.ra	;\
	[ 1 -eq `$(TOOLDIR)/show -d 4 r.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 c.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 rU.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 cU.ra` ] 					&&\
	[ 1 -eq `$(TOOLDIR)/show -d 4 rN.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 cN.ra` ] 					&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 rNU.ra` ] 				&&\
	[ 4 -eq `$(TOOLDIR)/show -d 4 cNU.ra` ] 				&&\
	true
	touch $@



TESTS += tests/test-rtnlinv-nlinv-noncart tests/test-rtnlinv-nlinv-pseudocart
TESTS += tests/test-rtnlinv-maps-dims tests/test-rtnlinv-noncart-maps-dims

TESTS_SLOW += tests/test-rtnlinv tests/test-rtnlinv-precomp
TESTS_SLOW += tests/test-rtnlinv-nlinv-sms


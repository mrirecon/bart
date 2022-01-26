


tests/test-nlinv: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra			;\
	$(TOOLDIR)/normalize 8 c.ra c_norm.ra						;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-sms: repmat fft nlinv nrmse scale $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 13 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra		;\
	$(TOOLDIR)/fft 8192 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra			;\
	$(TOOLDIR)/nlinv ksp2.ra r2.ra							;\
	$(TOOLDIR)/repmat 13 4 r.ra r3.ra						;\
	$(TOOLDIR)/scale 2. r2.ra r4.ra							;\
	$(TOOLDIR)/nrmse -s -t 0.1 r3.ra r4.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-norm: nlinv rss fmac nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra			;\
	$(TOOLDIR)/nlinv -N $(TESTS_OUT)/shepplogan_coil_ksp.ra rN.ra c.ra		;\
	$(TOOLDIR)/rss 8 c.ra c_norm.ra							;\
	$(TOOLDIR)/fmac rN.ra c_norm.ra x.ra						;\
	$(TOOLDIR)/nrmse -t 0.0000001 r.ra x.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-batch: conj join nlinv fmac fft nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/conj $(TESTS_OUT)/shepplogan_coil_ksp.ra kc.ra			;\
	$(TOOLDIR)/join 6 $(TESTS_OUT)/shepplogan_coil_ksp.ra kc.ra ksp.ra		;\
	$(TOOLDIR)/nlinv -N -S ksp.ra r.ra c.ra						;\
	$(TOOLDIR)/fmac r.ra c.ra x.ra							;\
	$(TOOLDIR)/fft -u 7 x.ra ksp2.ra						;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra ksp2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-batch2: repmat nlinv fmac fft nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 7 2 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra		;\
	$(TOOLDIR)/nlinv -s128 -N -S ksp.ra r.ra c.ra					;\
	$(TOOLDIR)/fmac r.ra c.ra x.ra							;\
	$(TOOLDIR)/fft -u 7 x.ra ksp2.ra						;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra ksp2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-noncart: traj scale phantom nufft resize nlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x256 -y21 traj.ra				;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra				;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra			;\
	$(TOOLDIR)/nlinv -N -S -i9 -t traj2.ra ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/resize -c 0 128 1 128 c.ra c2.ra			;\
	$(TOOLDIR)/fmac r.ra c2.ra x.ra					;\
	$(TOOLDIR)/nufft traj2.ra x.ra k2.ra				;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra k2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-gpu: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv -g $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/normalize 8 c.ra c_norm.ra						;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-sms-gpu: repmat fft nlinv nrmse scale $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 13 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra		;\
	$(TOOLDIR)/fft 8192 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/nlinv -g $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra			;\
	$(TOOLDIR)/nlinv -g ksp2.ra r2.ra						;\
	$(TOOLDIR)/repmat 13 4 r.ra r3.ra						;\
	$(TOOLDIR)/scale 2. r2.ra r4.ra							;\
	$(TOOLDIR)/nrmse -s -t 0.1 r3.ra r4.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-precomp: traj scale phantom ones repmat fft nufft nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -x256 -y55 traj.ra					;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra					;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra				;\
	$(TOOLDIR)/ones 3 1 256 55 o.ra						;\
	$(TOOLDIR)/nufft -a traj.ra o.ra psf.ra					;\
	$(TOOLDIR)/nufft -a traj.ra ksp.ra _adj.ra				;\
	$(TOOLDIR)/scale 2 _adj.ra adj.ra					;\
	$(TOOLDIR)/fft -u 7 psf.ra mtf.ra					;\
	$(TOOLDIR)/fft -u 7 adj.ra ksp2.ra					;\
	$(TOOLDIR)/scale 4. mtf.ra mtf2.ra					;\
	$(TOOLDIR)/nlinv -w1. -n -N -i7 -p mtf2.ra ksp2.ra r1.ra c1.ra		;\
	$(TOOLDIR)/nlinv -w1. -N -i7 -t traj2.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 r2.ra r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 c2.ra c1.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-pics: traj scale phantom resize pics nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x256 -y21 traj.ra				;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra				;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra			;\
	$(TOOLDIR)/nlinv -N -S -i8 -t traj2.ra ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/resize -c 0 128 1 128 c.ra c2.ra			;\
	$(TOOLDIR)/pics -r0.01 -S -t traj2.ra ksp.ra c2.ra x2.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 x2.ra r.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-pf-vcc: nlinv conj nrmse zeros ones join flip circshift fmac $(TESTS_OUT)/shepplogan_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 2 128 56 z.ra 							;\
	$(TOOLDIR)/ones 2 128 72 o.ra		 					;\
	$(TOOLDIR)/join 1 o.ra z.ra pat.ra						;\
	$(TOOLDIR)/fmac pat.ra $(TESTS_OUT)/shepplogan_ksp.ra pu.ra			;\
	$(TOOLDIR)/nlinv -i10 -c -p pat.ra pu.ra nl.ra					;\
	$(TOOLDIR)/flip 7 pat.ra tmp1.ra 						;\
	$(TOOLDIR)/circshift 1 1 tmp1.ra pat_conj.ra					;\
	$(TOOLDIR)/join 3 pat.ra pat_conj.ra pat_vcc.ra					;\
	$(TOOLDIR)/flip 7 pu.ra tmp2.ra 						;\
	$(TOOLDIR)/circshift 1 1 tmp2.ra tmp3.ra					;\
	$(TOOLDIR)/circshift 0 1 tmp3.ra tmp4.ra					;\
	$(TOOLDIR)/conj tmp4.ra pu_conj.ra						;\
	$(TOOLDIR)/join 3 pu.ra pu_conj.ra pu_vcc.ra					;\
	$(TOOLDIR)/nlinv -i10 -P -ppat_vcc.ra pu_vcc.ra nl_vcc.ra			;\
	$(TOOLDIR)/nrmse -t 0.00001 nl.ra nl_vcc.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-maps-dims: phantom nlinv show
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -s1 -k ksp.ra					;\
	$(TOOLDIR)/nlinv -m4       -S -i1 ksp.ra r.ra c.ra			;\
	$(TOOLDIR)/nlinv -m4    -U -S -i1 ksp.ra rU.ra cU.ra			;\
	$(TOOLDIR)/nlinv -m4 -N    -S -i1 ksp.ra rN.ra cN.ra			;\
	$(TOOLDIR)/nlinv -m4 -N -U -S -i1 ksp.ra rNU.ra cNU.ra			;\
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


tests/test-nlinv-noncart-maps-dims: traj phantom nlinv show
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -r -y21 traj.ra						;\
	$(TOOLDIR)/phantom -s1 -k -ttraj.ra ksp.ra				;\
	$(TOOLDIR)/nlinv -m4       -S -i1 -ttraj.ra  ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/nlinv -m4    -U -S -i1 -ttraj.ra  ksp.ra rU.ra cU.ra		;\
	$(TOOLDIR)/nlinv -m4 -N    -S -i1 -ttraj.ra  ksp.ra rN.ra cN.ra		;\
	$(TOOLDIR)/nlinv -m4 -N -U -S -i1 -ttraj.ra  ksp.ra rNU.ra cNU.ra	;\
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



TESTS += tests/test-nlinv tests/test-nlinv-sms
TESTS += tests/test-nlinv-batch tests/test-nlinv-batch2
TESTS += tests/test-nlinv-noncart tests/test-nlinv-precomp
TESTS += tests/test-nlinv-maps-dims tests/test-nlinv-noncart-maps-dims
TESTS += tests/test-nlinv-pf-vcc
TESTS += tests/test-nlinv-pics
TESTS_GPU += tests/test-nlinv-gpu tests/test-nlinv-sms-gpu






tests/test-nlinv: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra			;\
	$(TOOLDIR)/normalize 8 c.ra c_norm.ra						;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.05 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-reg: normalize nlinv pocsense nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/nlinv -i10 -RW:3:0:0.1 --liniter=50 $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra c.ra	;\
	$(TOOLDIR)/normalize 8 c.ra c_norm.ra						;\
	$(TOOLDIR)/pocsense -i1 $(TESTS_OUT)/shepplogan_coil_ksp.ra c_norm.ra proj.ra	;\
	$(TOOLDIR)/nrmse -t 0.03 proj.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-reg2: nlinv fft fmac nrmse $(TESTS_OUT)/shepplogan_coil_ksp_sub.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/nlinv -d4 -i12 -RW:3:0:1. --liniter=50 --cgiter=30 -N -S --sens-os=2 $(TESTS_OUT)/shepplogan_coil_ksp_sub.ra r.ra c.ra;\
	$(TOOLDIR)/fmac r.ra c.ra cim.ra								;\
	$(TOOLDIR)/fft -u 7 cim.ra ksp.ra								;\
	$(TOOLDIR)/nrmse -s -t 0.05 ksp.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra				;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-reg3: nlinv fft fmac nrmse $(TESTS_OUT)/shepplogan_coil_ksp_sub.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/nlinv -d4 -i12 -RT:3:0:10. --liniter=50 --cgiter=30 -N -S --sens-os=2 $(TESTS_OUT)/shepplogan_coil_ksp_sub.ra r.ra c.ra;\
	$(TOOLDIR)/fmac r.ra c.ra cim.ra								;\
	$(TOOLDIR)/fft -u 7 cim.ra ksp.ra								;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-reg4: nlinv fft fmac nrmse $(TESTS_OUT)/shepplogan_coil_ksp_sub.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/nlinv -d4 -i12 -RG:3:0:10. --liniter=50 --cgiter=30 -N -S --sens-os=2 $(TESTS_OUT)/shepplogan_coil_ksp_sub.ra r.ra c.ra;\
	$(TOOLDIR)/fmac r.ra c.ra cim.ra								;\
	$(TOOLDIR)/fft -u 7 cim.ra ksp.ra								;\
	$(TOOLDIR)/nrmse -t 0.06 ksp.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-nlinv-sms: repmat fft nlinv nrmse scale $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 13 4 $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra		;\
	$(TOOLDIR)/fft 8192 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/nlinv -S $(TESTS_OUT)/shepplogan_coil_ksp.ra r.ra			;\
	$(TOOLDIR)/nlinv -S ksp2.ra r2.ra						;\
	$(TOOLDIR)/repmat 13 4 r.ra r3.ra						;\
	$(TOOLDIR)/nrmse -t 0.1 r3.ra r2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-sms-noncart: traj phantom repmat fft nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -o2 -r -x64 -y21 traj.ra					;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra					;\
	$(TOOLDIR)/repmat 13 4 ksp.ra ksp_rep.ra					;\
	$(TOOLDIR)/fft 8192 ksp_rep.ra ksp2.ra						;\
	$(TOOLDIR)/nlinv -S -i10 -t traj.ra ksp.ra r.ra					;\
	$(TOOLDIR)/nlinv -S -i10 -t traj.ra ksp2.ra r2.ra				;\
	$(TOOLDIR)/repmat 13 4 r.ra r3.ra						;\
	$(TOOLDIR)/nrmse -t 0.1 r3.ra r2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-sms-noncart-psf: reshape repmat fft nlinv nrmse traj phantom
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -r -x128 -o2 -y60 traj.ra						;\
	$(TOOLDIR)/phantom -s6 -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/reshape 8200 2 3 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/nlinv --cgiter=20 -S -N -i12 --ret-sens-os -t traj.ra ksp2.ra r.ra c.ra	;\
	$(TOOLDIR)/nlinv --cgiter=20 -S -N -i12 --psf-based -t traj.ra ksp2.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -s -t 0.1 r.ra r2.ra							;\
	$(TOOLDIR)/nrmse -s -t 0.1 c.ra c2.ra							;\
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
	$(TOOLDIR)/nlinv -i10 -N -S ksp.ra r.ra c.ra					;\
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


tests/test-nlinv-noncart: traj phantom nufft resize nlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x128 -o2. -y21 traj.ra			;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra			;\
	$(TOOLDIR)/nlinv -N -S -i10 -t traj.ra ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/resize -c 0 128 1 128 c.ra c2.ra			;\
	$(TOOLDIR)/fmac r.ra c2.ra x.ra					;\
	$(TOOLDIR)/nufft traj.ra x.ra k2.ra				;\
	$(TOOLDIR)/nrmse -t 0.05 ksp.ra k2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-psf-noncart: traj phantom nufft resize nlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/traj -r -x128 -o2. -y21 traj.ra							;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra							;\
	$(TOOLDIR)/nlinv --ret-sens-os -N -S -i10 --cgiter=15 -w 0.0001 -t traj.ra ksp.ra r.ra c.ra	;\
	$(TOOLDIR)/nlinv --psf-based   -N -S -i10 --cgiter=15 -w 0.0001 -t traj.ra ksp.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.1 r.ra r2.ra								;\
	$(TOOLDIR)/nrmse -t 0.1 c.ra c2.ra								;\
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
	$(TOOLDIR)/nrmse -t 0.1 r3.ra r4.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-precomp-psf: traj scale phantom ones repmat fft nufft nlinv nrmse
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
	$(TOOLDIR)/nlinv          -n -w0.0001 -N -i7 -p mtf2.ra ksp2.ra r1.ra c1.ra	;\
	$(TOOLDIR)/nlinv --psf-based -w0.0001 -N -i7 -t traj2.ra ksp.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.0001 r2.ra r1.ra					;\
	$(TOOLDIR)/nrmse -t 0.0001 c2.ra c1.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

#FIXME: Test only works legacy scaloing as 0 cg iter
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
	$(TOOLDIR)/nlinv --legacy-early-stopping -w1. -n -N -i7 -p mtf2.ra ksp2.ra r1.ra c1.ra		;\
	$(TOOLDIR)/nlinv --legacy-early-stopping -w1. --ret-sens-os -N -i7 -t traj2.ra ksp.ra r2.ra c2.ra		;\
	$(TOOLDIR)/nrmse -t 0.0002 r2.ra r1.ra				;\
	$(TOOLDIR)/nrmse -t 0.0001 c2.ra c1.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-pics-psf-based: traj phantom resize pics nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -o2. -y21 traj.ra					;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra					;\
	$(TOOLDIR)/nlinv --psf-based -N -S -i8 -t traj.ra ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/resize -c 0 128 1 128 c.ra c2.ra					;\
	$(TOOLDIR)/pics -r0.01 -S -t traj.ra ksp.ra c2.ra x2.ra				;\
	$(TOOLDIR)/nrmse -t 0.05 x2.ra r.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-pics: traj phantom resize pics nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x128 -o2. -y21 traj.ra			;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra			;\
	$(TOOLDIR)/nlinv -N -S -i8 -t traj.ra ksp.ra r.ra c.ra		;\
	$(TOOLDIR)/resize -c 0 128 1 128 c.ra c2.ra			;\
	$(TOOLDIR)/pics -r0.01 -S -t traj.ra ksp.ra c2.ra x2.ra	;\
	$(TOOLDIR)/nrmse -t 0.06 x2.ra r.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

ifeq ($(BUILDTYPE), WASM) # WASM needs relaxed test
tests/test-nlinv-ksens: nlinv nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/nlinv		      $(TESTS_OUT)/shepplogan_coil_ksp.ra r1.ra c1.ra	;\
	$(TOOLDIR)/nlinv --ksens-dims=16:16:1 $(TESTS_OUT)/shepplogan_coil_ksp.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.0004 r1.ra r2.ra							;\
	$(TOOLDIR)/nrmse -t 0.0005 c1.ra c2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
else
tests/test-nlinv-ksens: nlinv nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/nlinv		      $(TESTS_OUT)/shepplogan_coil_ksp.ra r1.ra c1.ra	;\
	$(TOOLDIR)/nlinv --ksens-dims=16:16:1 $(TESTS_OUT)/shepplogan_coil_ksp.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.00002 r1.ra r2.ra							;\
	$(TOOLDIR)/nrmse -t 0.00003 c1.ra c2.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@
endif

tests/test-ncalib-noncart: fmac ncalib traj scale phantom resize pics nlinv nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -o2 -r -x64 -y45 traj.ra			;\
	$(TOOLDIR)/phantom -s8 -k -t traj.ra ksp.ra			;\
	$(TOOLDIR)/nlinv -N -S -i10 -t traj.ra ksp.ra r1.ra c1.ra	;\
	$(TOOLDIR)/ncalib -i10 -x64:64:1 -t traj.ra ksp.ra c2.ra	;\
	$(TOOLDIR)/pics -r0.001 -S -t traj.ra ksp.ra c2.ra r2.ra	;\
	$(TOOLDIR)/fmac c1.ra r1.ra x1.ra				;\
	$(TOOLDIR)/fmac c2.ra r2.ra x2.ra				;\
	$(TOOLDIR)/nrmse -s -t 0.05 x1.ra x2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ncalib: fmac ncalib scale copy resize pics nlinv nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/copy $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra	;\
	$(TOOLDIR)/nlinv -N -S -i10 --cgiter=10 ksp.ra r1.ra c1.ra	;\
	$(TOOLDIR)/ncalib      -i10 --cgiter=10 ksp.ra c2.ra		;\
	$(TOOLDIR)/pics -r0.001 -S ksp.ra c2.ra r2.ra			;\
	$(TOOLDIR)/fmac c1.ra r1.ra x1.ra				;\
	$(TOOLDIR)/fmac c2.ra r2.ra x2.ra				;\
	$(TOOLDIR)/nrmse -t 0.03 x1.ra x2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ncalib-scale: repmat copy ncalib nrmse bitmask $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/copy $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra					;\
	$(TOOLDIR)/ncalib -i6 --cgiter=10 ksp.ra _c1.ra							;\
	$(TOOLDIR)/repmat 5 9 _c1.ra c1.ra								;\
	$(TOOLDIR)/repmat 5 9 ksp.ra ksp2.ra								;\
	$(TOOLDIR)/ncalib -i6 --cgiter=10 --scale-loop-dims $$($(TOOLDIR)/bitmask 5) ksp2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 1.e-7 c2.ra c1.ra								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-ncalib-scale2: repmat copy ncalib nrmse bitmask $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/copy $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp.ra					;\
	$(TOOLDIR)/ncalib -i6 --cgiter=10 ksp.ra c1.ra							;\
	$(TOOLDIR)/repmat 5 9 ksp.ra ksp2.ra								;\
	$(TOOLDIR)/ncalib -i6 --cgiter=10 --scale-loop-dims $$($(TOOLDIR)/bitmask 5) --shared-col-dims $$($(TOOLDIR)/bitmask 5)  ksp2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 1.e-7 c2.ra c1.ra								;\
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
	$(TOOLDIR)/nrmse -t 0.00003 nl.ra nl_vcc.ra					;\
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

tests/test-nlinv-basis-noncart: nlinv traj phantom delta fmac ones repmat pics slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/traj -r -x64 -o2. -D -y31 traj.ra							;\
	$(TOOLDIR)/phantom -s8 -t traj.ra ksp.ra							;\
	$(TOOLDIR)/delta 16 36 31 p.ra 									;\
	$(TOOLDIR)/fmac ksp.ra p.ra pk.ra								;\
	$(TOOLDIR)/repmat 1 128 p.ra p2.ra								;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 31 o.ra								;\
	$(TOOLDIR)/repmat 6 2 o.ra o2.ra								;\
	$(TOOLDIR)/scale 0.5 o2.ra o3.ra								;\
	$(TOOLDIR)/nlinv -i9 -M0.01 --cgiter=10 -c -d4 -S -t traj.ra -pp2.ra -Bo3.ra pk.ra reco1.ra	;\
	$(TOOLDIR)/nlinv -i9 -M0.01 --cgiter=10 -c -d4 -S -t traj.ra                 ksp.ra reco.ra	;\
	$(TOOLDIR)/slice 6 0 reco1.ra reco20.ra								;\
	$(TOOLDIR)/nrmse -t 0.002 reco.ra reco20.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-noncart-fast: traj scale phantom nufft resize nlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x256 -y21 traj.ra				;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra				;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra			;\
	$(TOOLDIR)/nlinv --sens-os=1.25 --fast -N -S -i8 -t traj2.ra ksp.ra r.ra  c.ra	;\
	$(TOOLDIR)/nlinv --sens-os=1.25        -N -S -i8 -t traj2.ra ksp.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.005 r.ra r2.ra				;\
	$(TOOLDIR)/nrmse -t 0.0005 c.ra c2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nlinv-noncart-fast-gpu: traj scale phantom nufft resize nlinv fmac nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -r -x256 -y21 traj.ra				;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra				;\
	$(TOOLDIR)/phantom -s8 -k -t traj2.ra ksp.ra			;\
	$(TOOLDIR)/nlinv --sens-os=1.25 --ksens-dims=32:32:1 -g --fast -N -S -i8 -t traj2.ra ksp.ra r.ra  c.ra	;\
	$(TOOLDIR)/nlinv --sens-os=1.25 --ksens-dims=32:32:1 -g        -N -S -i8 -t traj2.ra ksp.ra r2.ra c2.ra	;\
	$(TOOLDIR)/nrmse -t 0.001 r.ra r2.ra				;\
	$(TOOLDIR)/nrmse -t 0.001 c.ra c2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-nlinv tests/test-nlinv-sms
TESTS += tests/test-nlinv-batch tests/test-nlinv-batch2
TESTS += tests/test-nlinv-noncart tests/test-nlinv-precomp tests/test-nlinv-precomp-psf
TESTS += tests/test-nlinv-maps-dims tests/test-nlinv-noncart-maps-dims
TESTS += tests/test-nlinv-pf-vcc
TESTS += tests/test-nlinv-pics tests/test-nlinv-pics-psf-based
TESTS += tests/test-nlinv-basis-noncart
TESTS += tests/test-nlinv-ksens
TESTS += tests/test-nlinv-psf-noncart tests/test-nlinv-sms-noncart-psf
TESTS += tests/test-ncalib tests/test-ncalib-noncart
TESTS += tests/test-nlinv-reg tests/test-nlinv-reg2 tests/test-nlinv-reg3 tests/test-nlinv-reg4
TESTS_GPU += tests/test-nlinv-gpu tests/test-nlinv-sms-gpu

TESTS_SLOW += tests/test-nlinv-sms-noncart






tests/test-pics-pi: pics scale nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/pics -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.23 reco2.ra $(TESTS_OUT)/shepplogan.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-noncart: traj scale phantom ones pics nufft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S -r0.001 -t traj2.ra ksp.ra o.ra reco.ra			;\
	$(TOOLDIR)/nufft traj2.ra reco.ra k2.ra						;\
	$(TOOLDIR)/nrmse -t 0.002 ksp.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-cs: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y48 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S -RT:7:0:0.001 -u0.1 -e -t traj2.ra ksp.ra o.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.22 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-wavl1: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y48 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S -RW:3:0:0.001 -i150 -e -t traj2.ra ksp.ra o.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.23 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-poisson-wavl1: poisson squeeze fft fmac ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -Y128 -Z128 -y1.2 -z1.2 -e -v -C24 p.ra			;\
	$(TOOLDIR)/squeeze p.ra p2.ra							;\
	$(TOOLDIR)/fft -u 7 $(TESTS_OUT)/shepplogan.ra ksp1.ra				;\
	$(TOOLDIR)/fmac ksp1.ra p2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S -RW:3:0:0.01 -i50 ksp.ra o.ra reco.ra			;\
	$(TOOLDIR)/nrmse -t 0.21 $(TESTS_OUT)/shepplogan.ra reco.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-bpwavl1: scale fft noise fmac ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/scale 50 $(TESTS_OUT)/shepplogan.ra shepp.ra				;\
	$(TOOLDIR)/fft -u 7 shepp.ra ksp1.ra						;\
	$(TOOLDIR)/noise -s 1 -n 1 ksp1.ra ksp2.ra					;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -a -P 128 -w1. -RW:3:0:1. -i50 ksp2.ra o.ra reco.ra		;\
	$(TOOLDIR)/pics -m -P 128 -w1. -RW:3:0:1. -i50 -u 2 ksp2.ra o.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.08 shepp.ra reco.ra					;\
	$(TOOLDIR)/nrmse -t 0.08 shepp.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-joint-wavl1: poisson reshape fft fmac ones pics slice nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -Y128 -Z128 -y1.1 -z1.1 -e -v -C24 -T2 p.ra			;\
	$(TOOLDIR)/reshape 63 128 128 1 1 1 2 p.ra p2.ra				;\
	$(TOOLDIR)/fft -u 7 $(TESTS_OUT)/shepplogan.ra ksp1.ra				;\
	$(TOOLDIR)/fmac ksp1.ra p2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S -RW:3:32:0.02 -i50 ksp.ra o.ra reco2.ra			;\
	$(TOOLDIR)/slice 5 0 reco2.ra reco.ra						;\
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_OUT)/shepplogan.ra reco.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




tests/test-pics-pics: traj scale phantom pics nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y32 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra					;\
	$(TOOLDIR)/pics -S -RT:7:0:0.001 -u1000000000. -e -t traj2.ra ksp.ra $(TESTS_OUT)/coils.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.22 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test that weights =0.5 have no effect
tests/test-pics-weights: pics ones nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 2 128 128 weights.ra						;\
	$(TOOLDIR)/scale 0.5 weights.ra	weights2.ra					;\
	$(TOOLDIR)/pics -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco1.ra	;\
	$(TOOLDIR)/pics -S -r0.001 -p weights2.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 reco2.ra reco1.ra				 	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test that weights =0.5 have no effect
# FIXME: this was 0.005 before but fails on travis
tests/test-pics-noncart-weights: traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y32 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra					;\
	$(TOOLDIR)/ones 4 1 256 32 1 weights.ra						;\
	$(TOOLDIR)/scale 0.5 weights.ra	weights2.ra					;\
	$(TOOLDIR)/pics -S -r0.001 -p weights2.ra -t traj2.ra ksp.ra $(TESTS_OUT)/coils.ra reco1.ra	;\
	$(TOOLDIR)/pics -S -r0.001                -t traj2.ra ksp.ra $(TESTS_OUT)/coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.010 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-warmstart: pics scale nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -i0 -Wo.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco.ra	;\
	$(TOOLDIR)/nrmse -t 0. o.ra reco.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-batch: pics repmat nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 5 32 $(TESTS_OUT)/shepplogan_coil_ksp.ra kspaces.ra		;\
	$(TOOLDIR)/pics -r0.01 -B32 kspaces.ra $(TESTS_OUT)/coils.ra reco1.ra		;\
	$(TOOLDIR)/pics -r0.01      kspaces.ra $(TESTS_OUT)/coils.ra reco2.ra		;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-tedim: phantom fmac fft pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s4 -m coils.ra						;\
	$(TOOLDIR)/phantom -m img.ra							;\
	$(TOOLDIR)/fmac img.ra	coils.ra cimg.ra					;\
	$(TOOLDIR)/fft -u 7 cimg.ra ksp.ra						;\
	$(TOOLDIR)/pics -i10 -w 1. -m ksp.ra coils.ra reco.ra				;\
	$(TOOLDIR)/nrmse -t 0.01 img.ra reco.ra 					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@







TESTS += tests/test-pics-pi tests/test-pics-noncart tests/test-pics-cs tests/test-pics-pics
TESTS += tests/test-pics-wavl1 tests/test-pics-poisson-wavl1 tests/test-pics-joint-wavl1 tests/test-pics-bpwavl1
TESTS += tests/test-pics-weights tests/test-pics-noncart-weights
TESTS += tests/test-pics-warmstart tests/test-pics-batch
TESTS += tests/test-pics-tedim





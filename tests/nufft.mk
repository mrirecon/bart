

# compare with FFT on a Cartesian grid

tests/test-nudft-forward: traj nufft reshape nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fft.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x128 -y128 traj.ra						;\
	$(TOOLDIR)/nufft -s traj.ra $(TESTS_OUT)/shepplogan.ra shepplogan_ksp2.ra	;\
	$(TOOLDIR)/reshape 7 128 128 1 shepplogan_ksp2.ra shepplogan_ksp3.ra		;\
	$(TOOLDIR)/nrmse -t 0.0001 $(TESTS_OUT)/shepplogan_fft.ra shepplogan_ksp3.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# compare with FFT on a Cartesian grid

tests/test-nufft-forward: traj nufft reshape nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -x128 -y128 traj.ra						;\
	$(TOOLDIR)/nufft traj.ra $(TESTS_OUT)/shepplogan.ra shepplogan_ksp2.ra		;\
	$(TOOLDIR)/reshape 7 128 128 1 shepplogan_ksp2.ra shepplogan_ksp3.ra		;\
	$(TOOLDIR)/nrmse -t 0.01 $(TESTS_OUT)/shepplogan_fftu.ra shepplogan_ksp3.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# compare nufft and nufdt

tests/test-nufft-nudft: traj nufft reshape nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y12 traj.ra						;\
	$(TOOLDIR)/nufft traj.ra $(TESTS_OUT)/shepplogan.ra shepplogan_ksp1.ra		;\
	$(TOOLDIR)/nufft -s traj.ra $(TESTS_OUT)/shepplogan.ra shepplogan_ksp2.ra	;\
	$(TOOLDIR)/scale 128. shepplogan_ksp1.ra shepplogan_ksp3.ra			;\
	$(TOOLDIR)/nrmse -t 0.002 shepplogan_ksp2.ra shepplogan_ksp3.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test adjoint using definition

tests/test-nudft-adjoint: zeros noise reshape traj nufft fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 64 64 1 z.ra						;\
	$(TOOLDIR)/noise -s123 z.ra n1.ra						;\
	$(TOOLDIR)/noise -s321 z.ra n2b.ra						;\
	$(TOOLDIR)/reshape 7 1 64 64 n2b.ra n2.ra					;\
	$(TOOLDIR)/traj -r -x64 -y64 traj.ra						;\
	$(TOOLDIR)/nufft -s traj.ra n1.ra k.ra						;\
	$(TOOLDIR)/nufft -s -a traj.ra n2.ra x.ra					;\
	$(TOOLDIR)/fmac -C -s7 n1.ra x.ra s1.ra						;\
	$(TOOLDIR)/fmac -C -s7 k.ra n2.ra s2.ra						;\
	$(TOOLDIR)/nrmse -t 0.00001 s1.ra s2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test adjoint using definition

tests/test-nufft-adjoint: zeros noise reshape traj nufft fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 128 128 1 z.ra						;\
	$(TOOLDIR)/noise -s123 z.ra n1.ra						;\
	$(TOOLDIR)/noise -s321 z.ra n2b.ra						;\
	$(TOOLDIR)/reshape 7 1 128 128 n2b.ra n2.ra					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/nufft traj.ra n1.ra k.ra						;\
	$(TOOLDIR)/nufft -a traj.ra n2.ra x.ra						;\
	$(TOOLDIR)/fmac -C -s7 n1.ra x.ra s1.ra						;\
	$(TOOLDIR)/fmac -C -s7 k.ra n2.ra s2.ra						;\
	$(TOOLDIR)/nrmse -t 0.00001 s1.ra s2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test inverse using definition

tests/test-nufft-inverse: traj scale phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y201 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/nufft -i traj2.ra ksp.ra reco.ra					;\
	$(TOOLDIR)/nufft traj2.ra reco.ra k2.ra						;\
	$(TOOLDIR)/nrmse -t 0.001 ksp.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test toeplitz by comparing to non-toeplitz

tests/test-nufft-toeplitz: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/nufft -l1. -i    traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft -l1. -i -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.002 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-nufft-forward tests/test-nufft-adjoint tests/test-nufft-inverse tests/test-nufft-toeplitz
TESTS += tests/test-nufft-nudft tests/test-nudft-forward tests/test-nudft-adjoint




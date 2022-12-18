

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
	$(TOOLDIR)/nufft -P traj.ra $(TESTS_OUT)/shepplogan.ra shepplogan_ksp2.ra	;\
	$(TOOLDIR)/reshape 7 128 128 1 shepplogan_ksp2.ra shepplogan_ksp3.ra		;\
	$(TOOLDIR)/nrmse -t 0.00005 $(TESTS_OUT)/shepplogan_fftu.ra shepplogan_ksp3.ra	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# compare nufft and nufdt

tests/test-nufft-nudft: traj nufft scale reshape nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y12 traj.ra						;\
	$(TOOLDIR)/nufft -P traj.ra $(TESTS_OUT)/shepplogan.ra shepplogan_ksp1.ra	;\
	$(TOOLDIR)/nufft -s traj.ra $(TESTS_OUT)/shepplogan.ra shepplogan_ksp2.ra	;\
	$(TOOLDIR)/scale 128. shepplogan_ksp1.ra shepplogan_ksp3.ra			;\
	$(TOOLDIR)/nrmse -t 0.00002 shepplogan_ksp2.ra shepplogan_ksp3.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test adjoint using definition

tests/test-nudft-adjoint: zeros noise reshape traj nufft fmac nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 3 64 64 1 z.ra							;\
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



# test adjoint linearity
tests/test-nufft-adj-lin: traj rss extract phantom fmac saxpy nrmse nufft
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -y55 t.ra							;\
	$(TOOLDIR)/rss 1 t.ra w.ra							;\
	$(TOOLDIR)/extract 1  0  32 t.ra t0.ra						;\
	$(TOOLDIR)/extract 1 32  64 t.ra t1.ra						;\
	$(TOOLDIR)/extract 1 64  96 t.ra t2.ra						;\
	$(TOOLDIR)/extract 1 96 128 t.ra t3.ra						;\
	$(TOOLDIR)/phantom -t t.ra -k k.ra						;\
	$(TOOLDIR)/fmac k.ra w.ra kw.ra							;\
	$(TOOLDIR)/extract 1  0  32 kw.ra k0.ra						;\
	$(TOOLDIR)/extract 1 32  64 kw.ra k1.ra						;\
	$(TOOLDIR)/extract 1 64  96 kw.ra k2.ra						;\
	$(TOOLDIR)/extract 1 96 128 kw.ra k3.ra						;\
	$(TOOLDIR)/nufft -x128:128:1 -a t0.ra k0.ra x0.ra				;\
	$(TOOLDIR)/nufft -x128:128:1 -a t1.ra k1.ra x1.ra				;\
	$(TOOLDIR)/nufft -x128:128:1 -a t2.ra k2.ra x2.ra				;\
	$(TOOLDIR)/nufft -x128:128:1 -a t3.ra k3.ra x3.ra				;\
	$(TOOLDIR)/saxpy -- 1. x0.ra x1.ra xa.ra					;\
	$(TOOLDIR)/saxpy -- 1. x2.ra x3.ra xb.ra					;\
	$(TOOLDIR)/saxpy -- 1. xa.ra xb.ra x.ra						;\
	$(TOOLDIR)/nufft -x128:128:1 -a t.ra kw.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 r.ra x.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test inverse using definition

tests/test-nufft-inverse: traj scale phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y201 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/nufft -r -i traj2.ra ksp.ra reco.ra					;\
	$(TOOLDIR)/nufft traj2.ra reco.ra k2.ra						;\
	$(TOOLDIR)/nrmse -t 0.001 ksp.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-inverse2: traj scale phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y201 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/nufft -x128:130:1 -i traj2.ra ksp.ra reco.ra				;\
	$(TOOLDIR)/nufft traj2.ra reco.ra k2.ra						;\
	$(TOOLDIR)/nrmse -t 0.001 ksp.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-inverse3: traj scale phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y201 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/nufft -x128:128:1 -i traj2.ra ksp.ra reco.ra				;\
	$(TOOLDIR)/nufft traj2.ra reco.ra k2.ra						;\
	$(TOOLDIR)/nrmse -t 0.001 ksp.ra k2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test toeplitz by comparing to non-toeplitz

tests/test-nufft-toeplitz: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/nufft -P -l1. -i -r traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft -P -l1. -i -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.0015 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test batch mode

tests/test-nufft-batch: traj phantom repmat nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x64 -y11 traj.ra						;\
	$(TOOLDIR)/phantom -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/nufft -i -r traj.ra ksp.ra reco.ra					;\
	$(TOOLDIR)/repmat 3 2 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/nufft -i -r traj.ra ksp2.ra reco2.ra					;\
	$(TOOLDIR)/repmat 3 2 reco.ra reco1.ra						;\
	$(TOOLDIR)/nrmse -t 0.0015 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




tests/test-nufft-gpu-inverse: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/nufft -l1.    -i -r traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft -l1. -g -i -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.002 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-adjoint: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -s4 -t traj.ra ksp.ra					;\
	$(TOOLDIR)/nufft     -a -r traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft -g  -a -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-forward: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -s4 phan.ra							;\
	$(TOOLDIR)/nufft    -r traj.ra phan.ra ksp1.ra					;\
	$(TOOLDIR)/nufft -g -t traj.ra phan.ra ksp2.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 ksp1.ra ksp2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-inverse-lowmem: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/nufft -l1.    -i -r traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft --lowmem -l1. -g -i -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.002 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-adjoint-lowmem: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -s4 -t traj.ra ksp.ra					;\
	$(TOOLDIR)/nufft     -a -r traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft --lowmem -g  -a -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-forward-lowmem: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -s4 phan.ra							;\
	$(TOOLDIR)/nufft    -r traj.ra phan.ra ksp1.ra					;\
	$(TOOLDIR)/nufft --lowmem -g -t traj.ra phan.ra ksp2.ra					;\
	$(TOOLDIR)/nrmse -t 0.00001 ksp1.ra ksp2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-inverse-precomp: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -t traj.ra ksp.ra						;\
	$(TOOLDIR)/nufft -l1.    -i -r traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft --no-precomp -l1. -i -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.002 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-adjoint-precomp: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -k -s4 -t traj.ra ksp.ra					;\
	$(TOOLDIR)/nufft     -a -r traj.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft --no-precomp -g  -a -t traj.ra ksp.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-gpu-forward-precomp: traj phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom -s4 phan.ra							;\
	$(TOOLDIR)/nufft    -r traj.ra phan.ra ksp1.ra					;\
	$(TOOLDIR)/nufft --no-precomp -g -t traj.ra phan.ra ksp2.ra				;\
	$(TOOLDIR)/nrmse -t 0.00001 ksp1.ra ksp2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nufft-over: traj phantom resize nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/phantom img.ra							;\
	$(TOOLDIR)/resize -c 0 256 1 256 img.ra img2.ra					;\
	$(TOOLDIR)/nufft    -r traj.ra img.ra ksp1.ra					;\
	$(TOOLDIR)/nufft -1 -r traj.ra img2.ra ksp2.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ksp1.ra ksp2.ra					;\
	$(TOOLDIR)/nufft -a    traj.ra ksp1.ra reco1.ra					;\
	$(TOOLDIR)/nufft -a -1 -r traj.ra ksp1.ra reco2a.ra				;\
	$(TOOLDIR)/resize -c 0 128 1 128 reco2a.ra reco2.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test low-mem adjoin

tests/test-nufft-lowmem-adjoint: zeros noise traj nufft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 4 1 128 128 3 z.ra						;\
	$(TOOLDIR)/noise -s321 z.ra n2.ra						;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/nufft -a traj.ra n2.ra x1.ra						;\
	$(TOOLDIR)/nufft --lowmem -a traj.ra n2.ra x2.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 x1.ra x2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-no-precomp-adjoint: zeros noise traj nufft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 4 1 128 128 3 z.ra						;\
	$(TOOLDIR)/noise -s321 z.ra n2.ra						;\
	$(TOOLDIR)/traj -r -x128 -y128 traj.ra						;\
	$(TOOLDIR)/nufft -a traj.ra n2.ra x1.ra						;\
	$(TOOLDIR)/nufft --no-precomp -a traj.ra n2.ra x2.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 x1.ra x2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# test inverse using definition

tests/test-nufft-lowmem-inverse: traj scale phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y201 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/nufft -m5 -r -i traj2.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft -m5 --lowmem -r -i traj2.ra ksp.ra reco2.ra			;\
	$(TOOLDIR)/nrmse -t 0.00006 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-nufft-no-precomp-inverse: traj scale phantom nufft nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y201 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/nufft -m5 -r -i traj2.ra ksp.ra reco1.ra				;\
	$(TOOLDIR)/nufft -m5 --no-precomp -r -i traj2.ra ksp.ra reco2.ra			;\
	$(TOOLDIR)/nrmse -t 0.00006 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS += tests/test-nufft-forward tests/test-nufft-adjoint tests/test-nufft-inverse tests/test-nufft-toeplitz
TESTS += tests/test-nufft-nudft tests/test-nudft-forward tests/test-nudft-adjoint tests/test-nufft-adj-lin
TESTS += tests/test-nufft-batch tests/test-nufft-over
TESTS += tests/test-nufft-lowmem-adjoint tests/test-nufft-lowmem-inverse tests/test-nufft-no-precomp-adjoint tests/test-nufft-no-precomp-inverse
TESTS += tests/test-nufft-inverse2 tests/test-nufft-inverse3

TESTS_GPU += tests/test-nufft-gpu-inverse tests/test-nufft-gpu-adjoint tests/test-nufft-gpu-forward
TESTS_GPU += tests/test-nufft-gpu-inverse-lowmem tests/test-nufft-gpu-adjoint-lowmem tests/test-nufft-gpu-forward-lowmem
TESTS_GPU += tests/test-nufft-gpu-inverse-precomp tests/test-nufft-gpu-adjoint-precomp tests/test-nufft-gpu-forward-precomp



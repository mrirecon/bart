
# for math in lowmem test
SHELL=/bin/bash


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


tests/test-pics-wavl1-dau2: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y48 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S --wavelet=dau2 -RW:3:0:0.001 -i150 -e -t traj2.ra ksp.ra o.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.23 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-wavl1-haar: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y48 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S --wavelet=haar -RW:3:0:0.001 -i150 -e -t traj2.ra ksp.ra o.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.22 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-wavl1-cdf44: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y48 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S --wavelet=cdf44 -RW:3:0:0.001 -i150 -e -t traj2.ra ksp.ra o.ra reco.ra	;\
	$(TOOLDIR)/scale 128. reco.ra reco2.ra						;\
	$(TOOLDIR)/nrmse -t 0.235 reco2.ra $(TESTS_OUT)/shepplogan.ra			;\
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
	$(TOOLDIR)/nrmse -t 0.22 $(TESTS_OUT)/shepplogan.ra reco.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-bpwavl1: scale fft noise fmac ones upat squeeze pics saxpy vec nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/scale 50 $(TESTS_OUT)/shepplogan.ra shepp.ra				;\
	$(TOOLDIR)/fft -u 7 shepp.ra ksp1.ra						;\
	$(TOOLDIR)/noise -s 1 -n 1 ksp1.ra ksp2.ra					;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/upat -Y 128 -Z 128 -y 2 -z 2 pat.ra					;\
	$(TOOLDIR)/squeeze pat.ra pat.ra	 					;\
	$(TOOLDIR)/fmac ksp2.ra pat.ra ksp3.ra	 					;\
	$(TOOLDIR)/pics -a -P 64 -w1. -n -RW:3:0:1. -i50 ksp3.ra o.ra reco.ra		;\
	$(TOOLDIR)/pics -m -P 64 -w1. -n -RW:3:0:1. -i100 -u .5 ksp3.ra o.ra reco2.ra	;\
	$(TOOLDIR)/fft -u 3 reco.ra kreco.ra						;\
	$(TOOLDIR)/saxpy -- -1 kreco.ra ksp3.ra ereco.ra				;\
	$(TOOLDIR)/fmac ereco.ra pat.ra preco.ra					;\
	$(TOOLDIR)/fmac -C -s 3 preco.ra preco.ra e.ra					;\
	$(TOOLDIR)/fft -u 3 reco2.ra kreco2.ra						;\
	$(TOOLDIR)/saxpy -- -1 kreco2.ra ksp3.ra ereco2.ra				;\
	$(TOOLDIR)/fmac ereco2.ra pat.ra preco2.ra					;\
	$(TOOLDIR)/fmac -C -s 3 preco2.ra preco2.ra e2.ra				;\
	$(TOOLDIR)/vec 4096 f.ra 							;\
	$(TOOLDIR)/nrmse -t .001 f.ra e.ra						;\
	$(TOOLDIR)/nrmse -t .001 f.ra e2.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-bp-noncart: traj scale phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -e -S -RW:3:0:0.0001 -t traj2.ra ksp.ra o.ra reco1.ra		;\
	$(TOOLDIR)/pics -e -a -P0.1 -S -RW:3:0:0.0001 -t traj2.ra ksp.ra o.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.09 reco1.ra reco2.ra					;\
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
	$(TOOLDIR)/pics -r0.01 -L32 kspaces.ra $(TESTS_OUT)/coils.ra reco1.ra		;\
	$(TOOLDIR)/pics -r0.01      kspaces.ra $(TESTS_OUT)/coils.ra reco2.ra		;\
	$(TOOLDIR)/nrmse -t 0. reco1.ra reco2.ra					;\
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



tests/test-pics-basis: ones delta fmac pics nrmse repmat scale slice $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/pics -S $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra gold.ra	;\
	$(TOOLDIR)/delta 16 33 128 p.ra 						;\
	$(TOOLDIR)/fmac $(TESTS_OUT)/shepplogan_coil_ksp.ra p.ra pk.ra			;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 128 o.ra						;\
	$(TOOLDIR)/repmat 6 2 o.ra o2.ra						;\
	$(TOOLDIR)/pics -S -Bo2.ra pk.ra $(TESTS_OUT)/coils.ra reco.ra			;\
	$(TOOLDIR)/scale 2. reco.ra reco2.ra						;\
	$(TOOLDIR)/slice 6 0 reco2.ra reco20.ra						;\
	$(TOOLDIR)/nrmse -t 0.000001 gold.ra reco20.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




tests/test-pics-basis-noncart: traj scale phantom delta fmac ones repmat pics slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -D -y31 traj.ra					;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/delta 16 36 31 p.ra 							;\
	$(TOOLDIR)/fmac ksp.ra p.ra pk.ra						;\
	$(TOOLDIR)/repmat 1 256 p.ra p2.ra						;\
	$(TOOLDIR)/ones 6 1 1 1 1 1 31 o.ra						;\
	$(TOOLDIR)/repmat 6 2 o.ra o2.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 coils.ra						;\
	$(TOOLDIR)/pics -r0.001 -t traj2.ra -pp2.ra -Bo2.ra pk.ra coils.ra reco1.ra	;\
	$(TOOLDIR)/pics -r0.001 -t traj2.ra ksp.ra coils.ra reco.ra			;\
	$(TOOLDIR)/scale 2. reco1.ra reco2.ra						;\
	$(TOOLDIR)/slice 6 0 reco2.ra reco20.ra						;\
	$(TOOLDIR)/nrmse -t 0.002 reco.ra reco20.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-basis-noncart2: traj scale phantom delta fmac ones repmat pics slice nrmse bitmask
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -D -y31 traj.ra					;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/delta 16 36 31 p.ra 							;\
	$(TOOLDIR)/fmac ksp.ra p.ra pk.ra						;\
	$(TOOLDIR)/repmat 1 256 p.ra p2.ra						;\
	$(TOOLDIR)/ones 7 1 1 1 1 1 31 2 o.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 coils.ra						;\
	$(TOOLDIR)/pics -S -r0.001 -t traj2.ra -pp2.ra -Bo.ra pk.ra coils.ra reco.ra	;\
	$(TOOLDIR)/fmac -s $$($(TOOLDIR)/bitmask 6) reco.ra o.ra reco2.ra		;\
	$(TOOLDIR)/nufft traj2.ra reco2.ra k2.ra					;\
	$(TOOLDIR)/fmac k2.ra p2.ra pk2.ra						;\
	$(TOOLDIR)/nrmse -t 0.003 pk.ra pk2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-basis-noncart-memory: traj scale phantom ones join transpose pics slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -D -y31 traj.ra					;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 7 1 1 1 1 1 31 2 o.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 coils.ra						;\
	$(TOOLDIR)/transpose 2 5 traj2.ra traj3.ra					;\
	$(TOOLDIR)/transpose 2 5 ksp.ra ksp1.ra						;\
	$(TOOLDIR)/pics -r0.001 -t traj3.ra -Bo.ra ksp1.ra coils.ra reco1.ra		;\
	$(TOOLDIR)/pics -r0.001 -t traj2.ra ksp.ra coils.ra reco.ra			;\
	$(TOOLDIR)/scale 2. reco1.ra reco2.ra						;\
	$(TOOLDIR)/slice 6 0 reco2.ra reco20.ra						;\
	$(TOOLDIR)/nrmse -t 0.002 reco.ra reco20.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-basis-noncart-memory2: traj scale phantom ones join noise transpose fmac pics slice nrmse zeros
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -D -y301 traj.ra					;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/phantom p.ra								;\
	$(TOOLDIR)/scale 0.5 ksp.ra ksp2.ra						;\
	$(TOOLDIR)/zeros 7 1 1 1 1 1 301 2 o.ra						;\
	$(TOOLDIR)/noise o.ra o1.ra							;\
	$(TOOLDIR)/join 6 ksp.ra ksp2.ra ksp3.ra					;\
	$(TOOLDIR)/transpose 2 5 ksp3.ra ksp4.ra					;\
	$(TOOLDIR)/fmac -s 64 ksp4.ra o1.ra ksp5.ra					;\
	$(TOOLDIR)/ones 3 128 128 1 coils.ra						;\
	$(TOOLDIR)/transpose 2 5 traj2.ra traj3.ra					;\
	$(TOOLDIR)/pics -S -U -i100 -r0. -t traj3.ra -Bo1.ra ksp5.ra coils.ra reco1.ra	;\
	$(TOOLDIR)/slice 6 0 reco1.ra reco.ra						;\
	$(TOOLDIR)/slice 6 1 reco1.ra reco2.ra						;\
	$(TOOLDIR)/scale 2. reco2.ra reco3.ra						;\
	$(TOOLDIR)/nrmse -s -t 0.25 reco.ra p.ra					;\
	$(TOOLDIR)/nrmse -s -t 0.25 reco3.ra p.ra					;\
	$(TOOLDIR)/nrmse -t 0.06 reco.ra reco3.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-noncart-sms: traj slice phantom conj join fft flip pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -y55 -m2 -r t.ra						;\
	$(TOOLDIR)/slice 13 0 t.ra t0.ra 						;\
	$(TOOLDIR)/slice 13 1 t.ra t1.ra 						;\
	$(TOOLDIR)/phantom -t t0.ra -s8 -k k0.ra					;\
	$(TOOLDIR)/phantom -t t1.ra -s8 -k k1.ra					;\
	$(TOOLDIR)/conj k1.ra k1C.ra							;\
	$(TOOLDIR)/join 13 k0.ra k1C.ra k.ra						;\
	$(TOOLDIR)/fft -n 8192 k.ra kk.ra						;\
	$(TOOLDIR)/phantom -S8 s.ra							;\
	$(TOOLDIR)/flip 7 s.ra sF.ra							;\
	$(TOOLDIR)/conj sF.ra sFC.ra							;\
	$(TOOLDIR)/join 13 s.ra sFC.ra ss.ra 						;\
	$(TOOLDIR)/pics -t t.ra -M kk.ra ss.ra x.ra					;\
	$(TOOLDIR)/slice 13 0 x.ra x0.ra						;\
	$(TOOLDIR)/slice 13 1 x.ra x1.ra						;\
	$(TOOLDIR)/phantom -k rk.ra							;\
	$(TOOLDIR)/conj rk.ra rkc.ra							;\
	$(TOOLDIR)/fft -i 7 rk.ra r0.ra							;\
	$(TOOLDIR)/fft -i 7 rkc.ra r1.ra						;\
	$(TOOLDIR)/join 13 r0.ra r1.ra r.ra						;\
	$(TOOLDIR)/nrmse -s -t 0.15 r.ra x.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# Without limiting the number of threads, this takes a very long time. The process appears
# to sleep for most of it, so it seems to be parallelization overhead.
tests/test-pics-lowmem: traj phantom repmat ones pics nrmse
	set +e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	export OMP_NUM_THREADS=4							;\
	$(TOOLDIR)/traj -x 16 -y 3 -D -r t.ra						;\
	$(TOOLDIR)/phantom -tt.ra -k k0.ra						;\
	$(TOOLDIR)/repmat 5 200 k0.ra k1.ra						;\
	$(TOOLDIR)/repmat 6 200 k1.ra k2.ra						;\
	$(TOOLDIR)/ones 2 4 4 s.ra							;\
	if [ $(( 100 * `$(TOOLDIR)/tests/print_max_rss.sh $(TOOLDIR)/pics -U -tt.ra k2.ra s.ra r1.ra` )) -ge \
	      $(( 75 * `$(TOOLDIR)/tests/print_max_rss.sh $(TOOLDIR)/pics    -tt.ra k2.ra s.ra r2.ra` )) ] ; then \
	     	echo "-U/--lowmem failed to reduce memory usage!"			;\
		false									;\
	fi										;\
	$(TOOLDIR)/nrmse -t 0.0005 r1.ra r2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-psf: traj phantom pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -D -y15 -o2. traj.ra						;\
	$(TOOLDIR)/phantom -t traj.ra ksp.ra						;\
	$(TOOLDIR)/phantom -S1 o.ra							;\
	$(TOOLDIR)/pics -S -r0.001 --psf_export=p.ra -t traj.ra ksp.ra o.ra reco1.ra	;\
	$(TOOLDIR)/pics -S -r0.001 --psf_import=p.ra -t traj.ra ksp.ra o.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.006 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pics-tgv: phantom slice noise fft ones pics tgv slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s2 x.ra							;\
	$(TOOLDIR)/slice 3 0 x.ra x0.ra							;\
	$(TOOLDIR)/noise -n 1000000. x0.ra x0n.ra					;\
	$(TOOLDIR)/fft -u 3 x0n.ra k0n.ra						;\
	$(TOOLDIR)/ones 2 128 128 o.ra							;\
	$(TOOLDIR)/pics -w1. -i100 -u0.1 -S -RG:3:0:750. k0n.ra o.ra x.ra		;\
	$(TOOLDIR)/tgv 750 3 x0n.ra xg.ra						;\
	$(TOOLDIR)/slice 15 0 xg.ra xg0.ra						;\
	$(TOOLDIR)/nrmse -t 0.000001 xg0.ra x.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-tgv2: phantom slice noise fft ones pics tgv slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s2 x.ra							;\
	$(TOOLDIR)/slice 3 0 x.ra x0.ra							;\
	$(TOOLDIR)/noise -n 1000000. x0.ra x0n.ra					;\
	$(TOOLDIR)/fft -u 3 x0n.ra k0n.ra						;\
	$(TOOLDIR)/ones 2 128 128 o.ra							;\
	$(TOOLDIR)/pics -w1. -i100 -u0.1 -S -RG:3:0:375. -RG:3:0:375. k0n.ra o.ra x.ra	;\
	$(TOOLDIR)/tgv 750 3 x0n.ra xg.ra						;\
	$(TOOLDIR)/slice 15 0 xg.ra xg0.ra						;\
	$(TOOLDIR)/nrmse -t 0.01 xg0.ra x.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-noncart-lowmem: traj slice phantom conj join fft flip pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -y55 -r t0.ra							;\
	$(TOOLDIR)/traj -y55 -G -r t1.ra						;\
	$(TOOLDIR)/phantom    -t t0.ra -s8 -k k0.ra					;\
	$(TOOLDIR)/phantom -T -t t1.ra -s8 -k k1C.ra					;\
	$(TOOLDIR)/conj k1C.ra k1.ra							;\
	$(TOOLDIR)/phantom -S8 s0.ra							;\
	$(TOOLDIR)/conj s0.ra s1.ra							;\
	$(TOOLDIR)/join 8 k0.ra k1.ra k.ra						;\
	$(TOOLDIR)/join 8 s0.ra s1.ra s.ra						;\
	$(TOOLDIR)/join 8 t0.ra t1.ra t.ra						;\
	$(TOOLDIR)/pics 	 -i2 -t t.ra k.ra s.ra r1.ra				;\
	$(TOOLDIR)/pics --lowmem -i2 -t t.ra k.ra s.ra r2.ra				;\
	$(TOOLDIR)/nrmse -s -t 0.000002 r1.ra r2.ra					;\
	$(TOOLDIR)/pics 	    -t t.ra k.ra s.ra r1.ra				;\
	$(TOOLDIR)/pics --lowmem    -t t.ra k.ra s.ra r2.ra				;\
	$(TOOLDIR)/nrmse -s -t 0.005 r1.ra r2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-noncart-lowmem-stack0: traj slice phantom conj join fft flip pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -y55 -r t0.ra							;\
	$(TOOLDIR)/phantom -t t0.ra -s8 -k k0.ra					;\
	$(TOOLDIR)/phantom -S8 s0.ra							;\
	$(TOOLDIR)/pics -r0.01			  -i2 -t t0.ra k0.ra s0.ra r1.ra	;\
	$(TOOLDIR)/pics -r0.01 --lowmem-stack=256 -i2 -t t0.ra k0.ra s0.ra r2.ra	;\
	$(TOOLDIR)/nrmse -s -t 0.000005 r1.ra r2.ra					;\
	$(TOOLDIR)/pics -r0.01			  -t t0.ra k0.ra s0.ra r1.ra		;\
	$(TOOLDIR)/pics -r0.01 --lowmem-stack=256 -t t0.ra k0.ra s0.ra r2.ra		;\
	$(TOOLDIR)/nrmse -s -t 0.005 r1.ra r2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-noncart-lowmem-no-toeplitz: traj slice phantom conj join fft flip pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/traj -y55 -r t0.ra								;\
	$(TOOLDIR)/phantom -t t0.ra -s8 -k k0.ra						;\
	$(TOOLDIR)/phantom -S8 s0.ra								;\
	$(TOOLDIR)/pics -r0.01 					-i2 -t t0.ra k0.ra s0.ra r1.ra	;\
	$(TOOLDIR)/pics -r0.01 --lowmem-stack=256 --no-toeplitz -i2 -t t0.ra k0.ra s0.ra r2.ra	;\
	$(TOOLDIR)/nrmse -s -t 0.00005 r1.ra r2.ra						;\
	$(TOOLDIR)/pics -r0.01 					-t t0.ra k0.ra s0.ra r1.ra	;\
	$(TOOLDIR)/pics -r0.01 --lowmem-stack=256 --no-toeplitz -t t0.ra k0.ra s0.ra r2.ra	;\
	$(TOOLDIR)/nrmse -s -t 0.005 r1.ra r2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-noncart-lowmem-stack1: traj slice phantom conj join fft flip pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -y55 -r t0.ra							;\
	$(TOOLDIR)/traj -y55 -G -r t1.ra						;\
	$(TOOLDIR)/phantom    -t t0.ra -s8 -k k0.ra					;\
	$(TOOLDIR)/phantom -T -t t1.ra -s8 -k k1C.ra					;\
	$(TOOLDIR)/conj k1C.ra k1.ra							;\
	$(TOOLDIR)/phantom -S8 s0.ra							;\
	$(TOOLDIR)/conj s0.ra s1.ra							;\
	$(TOOLDIR)/join 8 k0.ra k1.ra k.ra						;\
	$(TOOLDIR)/join 8 s0.ra s1.ra s.ra						;\
	$(TOOLDIR)/join 8 t0.ra t1.ra t.ra						;\
	$(TOOLDIR)/pics 		   -i2 -t t.ra k.ra s.ra r1.ra			;\
	$(TOOLDIR)/pics --lowmem-stack=256 -i2 -t t.ra k.ra s.ra r2.ra			;\
	$(TOOLDIR)/nrmse -s -t 0.00001 r1.ra r2.ra					;\
	$(TOOLDIR)/pics 		   -i2 -t t.ra k.ra s.ra r1.ra			;\
	$(TOOLDIR)/pics --lowmem-stack=256 -i2 -t t.ra k.ra s.ra r2.ra			;\
	$(TOOLDIR)/nrmse -s -t 0.005 r1.ra r2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-noncart-lowmem-stack2: traj slice phantom conj join fft flip pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -y55 -r t0.ra							;\
	$(TOOLDIR)/traj -y55 -G -r t1.ra						;\
	$(TOOLDIR)/phantom    -t t0.ra -s8 -k k0.ra					;\
	$(TOOLDIR)/phantom -T -t t1.ra -s8 -k k1C.ra					;\
	$(TOOLDIR)/conj k1C.ra k1.ra							;\
	$(TOOLDIR)/phantom -S8 s0.ra							;\
	$(TOOLDIR)/conj s0.ra s1.ra							;\
	$(TOOLDIR)/join 8 k0.ra k1.ra k.ra						;\
	$(TOOLDIR)/join 8 s0.ra s1.ra s.ra						;\
	$(TOOLDIR)/join 8 t0.ra t1.ra t.ra						;\
	$(TOOLDIR)/pics 		   -i2 -t t.ra k.ra s.ra r1.ra			;\
	$(TOOLDIR)/pics --lowmem-stack=264 -i2 -t t.ra k.ra s.ra r2.ra			;\
	$(TOOLDIR)/nrmse -s -t 0.00005 r1.ra r2.ra					;\
	$(TOOLDIR)/pics 		   -t t.ra k.ra s.ra r1.ra			;\
	$(TOOLDIR)/pics --lowmem-stack=264 -t t.ra k.ra s.ra r2.ra			;\
	$(TOOLDIR)/nrmse -s -t 0.005 r1.ra r2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-pics-pi tests/test-pics-noncart tests/test-pics-cs tests/test-pics-pics
TESTS += tests/test-pics-poisson-wavl1 tests/test-pics-joint-wavl1 tests/test-pics-bpwavl1
TESTS += tests/test-pics-weights tests/test-pics-noncart-weights
TESTS += tests/test-pics-warmstart tests/test-pics-batch
TESTS += tests/test-pics-tedim tests/test-pics-bp-noncart
TESTS += tests/test-pics-basis tests/test-pics-basis-noncart tests/test-pics-basis-noncart-memory tests/test-pics-basis-noncart2
#TESTS += tests/test-pics-lowmem
TESTS += tests/test-pics-noncart-sms tests/test-pics-psf tests/test-pics-tgv tests/test-pics-tgv2
TESTS += tests/test-pics-wavl1-dau2 tests/test-pics-wavl1-cdf44 tests/test-pics-wavl1-haar
TESTS += tests/test-pics-noncart-lowmem tests/test-pics-noncart-lowmem-stack0 tests/test-pics-noncart-lowmem-stack1 tests/test-pics-noncart-lowmem-stack2 tests/test-pics-noncart-lowmem-no-toeplitz


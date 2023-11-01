
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
	$(TOOLDIR)/nrmse -t .002 f.ra e2.ra 						;\
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
tests/test-pics-weights: pics scale ones nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
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


tests/test-pics-warmstart: pics ones nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -i0 -Wo.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco.ra	;\
	$(TOOLDIR)/nrmse -t 0. o.ra reco.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-timedim: phantom fmac fft pics nrmse
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

tests/test-pics-basis-noncart2: traj scale phantom delta fmac ones repmat pics nufft nrmse bitmask
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

tests/test-pics-phase: traj phantom ones join nufft fmac scale bitmask saxpy slice pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x64 -o2 -y13 -t3 t.ra					;\
	$(TOOLDIR)/phantom -N2 -b -t t.ra -s8 -k k0.ra					;\
	$(TOOLDIR)/ones 1 1 one.ra							;\
	$(TOOLDIR)/scale -- -1. one.ra mone.ra						;\
	$(TOOLDIR)/scale -- -2. one.ra mtwo.ra						;\
	$(TOOLDIR)/scale 2. one.ra two.ra						;\
	$(TOOLDIR)/join 10 one.ra one.ra mone.ra scl1.ra				;\
	$(TOOLDIR)/join 10 two.ra mtwo.ra two.ra scl2.ra				;\
	$(TOOLDIR)/join 6 scl1.ra scl2.ra scl.ra					;\
	$(TOOLDIR)/fmac -s $$($(TOOLDIR)/bitmask 6) scl.ra k0.ra k.ra			;\
	$(TOOLDIR)/ones 2 64 64 phmap0.ra						;\
	$(TOOLDIR)/phantom -N2 -b -x64 i.ra						;\
	$(TOOLDIR)/fmac -s $$($(TOOLDIR)/bitmask 6) scl.ra i.ra ref1.ra			;\
	$(TOOLDIR)/slice 10 0 ref1.ra ref.ra						;\
	$(TOOLDIR)/slice 6 0 i.ra i1.ra							;\
	$(TOOLDIR)/slice 6 1 i.ra i2.ra							;\
	$(TOOLDIR)/saxpy -- -2 i2.ra phmap0.ra phmap1.ra				;\
	$(TOOLDIR)/saxpy -- -2 i1.ra phmap0.ra phmap2.ra				;\
	$(TOOLDIR)/join 10 phmap0.ra phmap1.ra phmap2.ra phmap.ra			;\
	$(TOOLDIR)/phantom -S8 -x64 c.ra						;\
	$(TOOLDIR)/fmac c.ra phmap.ra c2.ra						;\
	$(TOOLDIR)/pics -S --shared-img-dims=$$($(TOOLDIR)/bitmask 10) -tt.ra k.ra c2.ra i.ra		;\
	$(TOOLDIR)/nrmse -s -t0.16 i.ra ref.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


$(TESTS_OUT)/ksp_usamp_1.ra: phantom poisson squeeze fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 64 -k -s 2 ksp_1.ra					;\
	$(TOOLDIR)/poisson -Y 64 -y 2 -Z 64 -z 2 -C 32 -e tmp_poisson_1.ra		;\
	$(TOOLDIR)/squeeze tmp_poisson_1.ra poisson_1.ra				;\
	$(TOOLDIR)/fmac ksp_1.ra poisson_1.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/ksp_usamp_2.ra: phantom poisson squeeze fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -G -x 64 -k -s 2 ksp_2.ra				;\
	$(TOOLDIR)/poisson -Y 64 -y 2 -Z 64 -z 2 -C 32 -e tmp_poisson_2.ra		;\
	$(TOOLDIR)/squeeze tmp_poisson_2.ra poisson_2.ra				;\
	$(TOOLDIR)/fmac ksp_2.ra poisson_2.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/ksp_usamp_3.ra: phantom poisson squeeze fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -B -x 64 -k -s 2 ksp_3.ra				;\
	$(TOOLDIR)/poisson -Y 64 -y 2 -Z 64 -z 2 -C 32 -e tmp_poisson_3.ra		;\
	$(TOOLDIR)/squeeze tmp_poisson_3.ra poisson_3.ra				;\
	$(TOOLDIR)/fmac ksp_3.ra poisson_3.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/sens_1.ra: ecalib $(TESTS_OUT)/ksp_usamp_1.ra
	$(TOOLDIR)/ecalib -S $(TESTS_OUT)/ksp_usamp_1.ra $@

$(TESTS_OUT)/sens_2.ra: ecalib $(TESTS_OUT)/ksp_usamp_2.ra
	$(TOOLDIR)/ecalib -S $(TESTS_OUT)/ksp_usamp_2.ra $@

$(TESTS_OUT)/sens_3.ra: ecalib $(TESTS_OUT)/ksp_usamp_3.ra
	$(TOOLDIR)/ecalib -S $(TESTS_OUT)/ksp_usamp_3.ra $@

$(TESTS_OUT)/img_l2_1.ra: pics $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/sens_1.ra
	$(TOOLDIR)/pics -S -l2 -r 0.005 -i 3 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/sens_1.ra $@

$(TESTS_OUT)/img_l2_2.ra: pics $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/sens_2.ra
	$(TOOLDIR)/pics -S -l2 -r 0.005 -i 3 $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/sens_2.ra $@

$(TESTS_OUT)/img_l2_3.ra: pics $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_3.ra
	$(TOOLDIR)/pics -S -l2 -r 0.005 -i 3 $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_3.ra $@

tests/test-pics-cart-loop: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref			;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p				;\
	$(ROOTDIR)/bart -l 8192 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p						;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref img_l2_p										;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-loop_range: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref_03		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref_13						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_3.ra img_l2_ref_23									;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra img_l2_ref_02						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra img_l2_ref_01									;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p				;\
	$(ROOTDIR)/bart -l 8192 -s 1 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p13					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_13 img_l2_p13										;\
	$(ROOTDIR)/bart -l 8192 -s 0 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p03					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_03 img_l2_p03										;\
	$(ROOTDIR)/bart -l 8192 -s 2 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p23					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_23 img_l2_p23										;\
	$(ROOTDIR)/bart -l 8192 -s 0 -e 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p02					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_02 img_l2_p02										;\
	$(ROOTDIR)/bart -l 8192 -s 0 -e 1 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p01					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_01 img_l2_p01										;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-slice:  bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p				;\
	$(ROOTDIR)/bart -l 8192 -s 0 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_0						;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_1.ra img_l2_0									;\
	$(ROOTDIR)/bart -l 8192 -s 1 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_1						;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_2.ra img_l2_1									;\
	$(ROOTDIR)/bart -l 8192 -s 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_2						;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_3.ra img_l2_2									;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-cart-loop_range-omp:  bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref_03.ra			;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref_13.ra						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_3.ra img_l2_ref_23.ra									;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra img_l2_ref_02.ra						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra img_l2_ref_01.ra									;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p			;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p					;\
	OMP_NUM_THREADS=2 $(ROOTDIR)/bart -l 8192 -s 1 -e 3 -t 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p13			;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_13.ra img_l2_p13										;\
	OMP_NUM_THREADS=4 $(ROOTDIR)/bart -l 8192 -s 0 -e 3 -t 4 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p03			;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_03.ra img_l2_p03										;\
	OMP_NUM_THREADS=2 $(ROOTDIR)/bart -l 8192 -s 2 -e 3 -t 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p23			;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_23.ra img_l2_p23										;\
	OMP_NUM_THREADS=2 $(ROOTDIR)/bart -l 8192 -s 0 -e 2 -t 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p02			;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_02.ra img_l2_p02										;\
	OMP_NUM_THREADS=2 $(ROOTDIR)/bart -l 8192 -s 0 -e 1 -t 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p01			;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_01.ra img_l2_p01										;\
	rm *.ra *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-cart-batch-mpi: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref.ra		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p				;\
	mpirun -n 4 --allow-run-as-root $(ROOTDIR)/bart -p 8192 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref.ra img_l2_p										;\
	rm *.ra *.cfl *.hdr  ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-range-batch-mpi: bart  $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref_03.ra			;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref_13.ra						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_3.ra img_l2_ref_23.ra									;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra img_l2_ref_02.ra						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra img_l2_ref_01.ra									;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p			;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p					;\
	mpirun -n 2  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 0 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p03		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_03.ra img_l2_p03										;\
	mpirun -n 4  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 1 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p13		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_13.ra img_l2_p13										;\
	mpirun -n 4  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 2 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p23		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_23.ra img_l2_p23										;\
	mpirun -n 4  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 0 -e 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p02		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_02.ra img_l2_p02										;\
	mpirun -n 3  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 0 -e 1 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p01		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_01.ra img_l2_p01										;\
	rm *.ra *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-mpi: bart pics copy nrmse $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/sens_1.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(TOOLDIR)/copy $(TESTS_OUT)/ksp_usamp_1.ra shepplogan_coil_ksp									;\
	$(TOOLDIR)/copy $(TESTS_OUT)/sens_1.ra coils												;\
	mpirun -n 2 --allow-run-as-root $(ROOTDIR)/bart pics --mpi=8 -S -r0.001 shepplogan_coil_ksp coils reco					;\
	                                $(ROOTDIR)/bart pics         -S -r0.001 shepplogan_coil_ksp coils reco_ref				;\
	$(TOOLDIR)/nrmse -t 1e-5 reco_ref reco													;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-mpi_shared: bart pics copy nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(TOOLDIR)/copy $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_coil_ksp									;\
	$(TOOLDIR)/copy $(TESTS_OUT)/coils.ra coils												;\
	mpirun -n 2 --allow-run-as-root $(ROOTDIR)/bart -S pics --mpi=8 -S -r0.001 shepplogan_coil_ksp coils reco				;\
	                                $(ROOTDIR)/bart    pics         -S -r0.001 shepplogan_coil_ksp coils reco_ref				;\
	$(TOOLDIR)/nrmse -t 1e-5 reco_ref reco													;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-mpi-sharedcoil: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_1.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra sens_p										;\
	mpirun -n 3 --allow-run-as-root $(ROOTDIR)/bart -S pics --mpi=8 -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p		;\
					$(ROOTDIR)/bart -S pics         -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_ref		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref img_l2_p										;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-mpi-timedim: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(ROOTDIR)/bart phantom -s4 -m coils													;\
	$(ROOTDIR)/bart phantom -m img.ra													;\
	$(ROOTDIR)/bart fmac img.ra	coils cimg.ra												;\
	$(ROOTDIR)/bart fft -u 7 cimg.ra ksp													;\
	mpirun -n 2 --allow-run-as-root $(ROOTDIR)/bart -S pics --mpi=$(shell $(ROOTDIR)/bart bitmask 13) -i10 -w 1. -m ksp coils reco		;\
	$(ROOTDIR)/bart pics -i10 -w 1. -m ksp coils ref.ra											;\
	$(ROOTDIR)/bart nrmse -t 1e-5 ref.ra reco 												;\
	rm *.cfl *.hdr *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-mpi-multipledims: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)														;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p					;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p							;\
	mpirun --allow-run-as-root --host localhost:6 -n 6 $(ROOTDIR)/bart -S pics --mpi=8200 -i10 -S -l2 -r 0.005 ksp_usamp_p sens_p img_l2_p			;\
							   $(ROOTDIR)/bart -S pics            -i10 -S -l2 -r 0.005 ksp_usamp_p sens_p img_l2_ref		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref img_l2_p													;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-mpi-multipledims-sharedcoil: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_1.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)														;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p					;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/sens_1.ra sens_p													;\
	mpirun --allow-run-as-root --host localhost:6 -n 6 $(ROOTDIR)/bart -S pics --mpi=8200 -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p			;\
							   $(ROOTDIR)/bart -S pics            -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_ref		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref img_l2_p													;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-noncart-mpi: bart traj scale phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2						;\
	$(TOOLDIR)/phantom -s4 -t traj2 ksp					;\
	$(TOOLDIR)/phantom -S4 col							;\
	mpirun -n 2 --allow-run-as-root $(ROOTDIR)/bart -S pics --mpi=8 -S --fista -e -r0.001 -t traj2 ksp col reco1	;\
					$(ROOTDIR)/bart    pics         -S --fista -e -r0.001 -t traj2 ksp col reco2	;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1 reco2					;\
	rm *.cfl *ra *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# Very slow due to recomposing via mpi 0=> deactivated
tests/test-pics-noncart-mpi-gridH: bart traj scale phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2						;\
	$(TOOLDIR)/phantom -s4 -t traj2 ksp					;\
	$(TOOLDIR)/phantom -S4 col							;\
	mpirun -n 2 --allow-run-as-root $(ROOTDIR)/bart -S pics --no-toeplitz --mpi=8 -S --fista -e -r0.001 -t traj2 ksp col reco1	;\
					$(ROOTDIR)/bart    pics --no-toeplitz         -S --fista -e -r0.001 -t traj2 ksp col reco2	;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1 reco2					;\
	rm *.cfl *ra *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-slice-batch-mpi: bart  $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p				;\
	mpirun -n 1  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 0 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_0					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_1.ra img_l2_0									;\
	mpirun -n 2  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 1 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_1					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_2.ra img_l2_1									;\
	mpirun -n 3  --allow-run-as-root $(ROOTDIR)/bart -l 8192 -s 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_2					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_3.ra img_l2_2									;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-non-cart-batch-mpi: bart $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)										;\
	$(ROOTDIR)/bart traj -r -D -x 64 -y 32 tr										;\
	$(ROOTDIR)/bart scale 0.5 tr tr1											;\
	$(ROOTDIR)/bart scale 2 tr tr2												;\
	$(ROOTDIR)/bart phantom -ttr -k -s 2 ksp_rad										;\
	$(ROOTDIR)/bart phantom -ttr1 -k -G -s 2 ksp_rad_g									;\
	$(ROOTDIR)/bart phantom -ttr2 -k -B -s 2 ksp_rad_b									;\
	$(ROOTDIR)/bart pics -i 3 -t tr ksp_rad $(TESTS_OUT)/sens_1.ra pics_rad							;\
	$(ROOTDIR)/bart pics -i 3 -t tr1 ksp_rad_g $(TESTS_OUT)/sens_2.ra pics_rad_g						;\
	$(ROOTDIR)/bart pics -i 3 -t tr2 ksp_rad_b $(TESTS_OUT)/sens_3.ra pics_rad_b						;\
	$(ROOTDIR)/bart join 13 pics_rad pics_rad_g pics_rad_b pics_rad_ref							;\
	$(ROOTDIR)/bart join 13 ksp_rad ksp_rad_g ksp_rad_b ksp_rad_p								;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens1_p			;\
	$(ROOTDIR)/bart join 13 tr tr1 tr2 tr_p											;\
	mpirun -n 2 --allow-run-as-root $(ROOTDIR)/bart -p 8192 -e 3 pics -i 3 -ttr_p ksp_rad_p sens1_p pics_rad_p		;\
	$(ROOTDIR)/bart nrmse -t 5e-3 pics_rad_ref pics_rad_p									;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-pics-pi tests/test-pics-noncart tests/test-pics-cs tests/test-pics-pics
TESTS += tests/test-pics-poisson-wavl1 tests/test-pics-joint-wavl1 tests/test-pics-bpwavl1
TESTS += tests/test-pics-weights tests/test-pics-noncart-weights
TESTS += tests/test-pics-warmstart
TESTS += tests/test-pics-timedim tests/test-pics-bp-noncart
TESTS += tests/test-pics-basis tests/test-pics-basis-noncart tests/test-pics-basis-noncart-memory tests/test-pics-basis-noncart2
#TESTS += tests/test-pics-lowmem
TESTS += tests/test-pics-noncart-sms tests/test-pics-psf tests/test-pics-tgv tests/test-pics-tgv2
TESTS += tests/test-pics-wavl1-dau2 tests/test-pics-wavl1-cdf44 tests/test-pics-wavl1-haar
TESTS += tests/test-pics-noncart-lowmem tests/test-pics-noncart-lowmem-stack0 tests/test-pics-noncart-lowmem-stack1 tests/test-pics-noncart-lowmem-stack2 tests/test-pics-noncart-lowmem-no-toeplitz
TESTS += tests/test-pics-phase
TESTS += tests/test-pics-cart-loop tests/test-pics-cart-loop_range tests/test-pics-cart-slice

ifeq ($(MPI),1)
TESTS_SLOW += tests/test-pics-cart-batch-mpi tests/test-pics-non-cart-batch-mpi tests/test-pics-cart-slice-batch-mpi tests/test-pics-cart-range-batch-mpi
TESTS_SLOW += tests/test-pics-cart-mpi tests/test-pics-noncart-mpi tests/test-pics-mpi-timedim
TESTS_SLOW += tests/test-pics-cart-mpi_shared tests/test-pics-cart-mpi-sharedcoil tests/test-pics-cart-mpi-multipledims tests/test-pics-cart-mpi-multipledims-sharedcoil
endif

ifeq ($(OMP),1)
TESTS_SLOW += tests/test-pics-cart-loop_range-omp
endif


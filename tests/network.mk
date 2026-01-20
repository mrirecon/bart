
# for math in lowmem test
SHELL := /bin/bash

PAT=$(TESTS_OUT)/pattern.ra
WGH=$(TESTS_OUT)/weights.ra
TRN_PAT=$(TESTS_OUT)/pattern_batch.ra
TRN_REF_KSP=$(TESTS_OUT)/train_ref_ksp.ra
TRN_REF_CIM=$(TESTS_OUT)/train_ref_cim.ra
TRN_REF_IMG=$(TESTS_OUT)/train_ref_img.ra
TRN_COL=$(TESTS_OUT)/train_sens.ra
TRN_KSP=$(TESTS_OUT)/train_ksp.ra
TRN_TRJ=$(TESTS_OUT)/train_trj.ra
TRN_KSP_NC=$(TESTS_OUT)/train_ksp_nc.ra
TRN_REF_KSP_NC=$(TESTS_OUT)/train_ksp_nc_ref.ra
TRN_MSK=$(TESTS_OUT)/train_mask.ra

TST_PAT=$(TESTS_OUT)/pattern_batch_tst.ra
TST_REF_KSP=$(TESTS_OUT)/tst_ref_ksp.ra
TST_REF_CIM=$(TESTS_OUT)/tst_ref_cim.ra
TST_REF_IMG=$(TESTS_OUT)/tst_ref_img.ra
TST_COL=$(TESTS_OUT)/tst_sens.ra
TST_KSP=$(TESTS_OUT)/tst_ksp.ra
TST_TRJ=$(TESTS_OUT)/tst_trj.ra
TST_KSP_NC=$(TESTS_OUT)/tst_ksp_nc.ra
TST_REF_KSP_NC=$(TESTS_OUT)/tst_ksp_nc_ref.ra

$(PAT): poisson reshape
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e $(TESTS_TMP)/poisson.ra	;\
	$(TOOLDIR)/reshape 7 32 32 1 poisson.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(WGH): poisson reshape
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e $(TESTS_TMP)/poisson.ra	;\
	$(TOOLDIR)/reshape 15 1 32 32 1 poisson.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TRN_PAT): poisson reshape join
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -s1 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e p1.ra			;\
	$(TOOLDIR)/poisson -s2 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e p2.ra			;\
	$(TOOLDIR)/poisson -s3 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e p3.ra			;\
	$(TOOLDIR)/poisson -s4 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e p4.ra			;\
	$(TOOLDIR)/poisson -s5 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e p5.ra			;\
	$(TOOLDIR)/reshape 7 32 32 1 p1.ra rp1.ra					;\
	$(TOOLDIR)/reshape 7 32 32 1 p2.ra rp2.ra					;\
	$(TOOLDIR)/reshape 7 32 32 1 p3.ra rp3.ra					;\
	$(TOOLDIR)/reshape 7 32 32 1 p4.ra rp4.ra					;\
	$(TOOLDIR)/reshape 7 32 32 1 p5.ra rp5.ra					;\
	$(TOOLDIR)/join 15 rp1.ra rp2.ra rp3.ra rp4.ra rp5.ra $@			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TRN_REF_KSP): phantom join scale
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r1 -k kphan1.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r2 -k kphan2.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r3 -k kphan3.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r4 -k kphan4.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r5 -k kphan5.ra				;\
	$(TOOLDIR)/join 15 kphan1.ra kphan2.ra kphan3.ra kphan4.ra kphan5.ra kphan.ra	;\
	$(TOOLDIR)/scale 0.00032 kphan.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TRN_REF_CIM): $(TRN_REF_KSP) fft
	$(TOOLDIR)/fft -i -u 7 $(TRN_REF_KSP) $@

$(TRN_REF_IMG): $(TRN_REF_CIM) $(TRN_COL) fmac
	$(TOOLDIR)/fmac -s8 -C $(TRN_REF_CIM) $(TRN_COL) $@

$(TRN_KSP): fmac $(TRN_PAT) $(TRN_REF_KSP)
	$(TOOLDIR)/fmac $(TRN_REF_KSP) $(TRN_PAT) $@

$(TRN_COL): phantom normalize repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/repmat 15 5 sens.ra sens.ra						;\
	$(TOOLDIR)/normalize 8 sens.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TRN_TRJ): traj
	$(TOOLDIR)/traj -x32 -y32 $@

$(TRN_KSP_NC): reshape $(TRN_KSP)
	$(TOOLDIR)/reshape 7 1 32 32 $(TRN_KSP) $@

$(TRN_REF_KSP_NC): reshape $(TRN_REF_KSP)
	$(TOOLDIR)/reshape 7 1 32 32 $(TRN_REF_KSP) $@



$(TST_PAT): poisson reshape join
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -s6 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e p1.ra			;\
	$(TOOLDIR)/poisson -s7 -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e p2.ra			;\
	$(TOOLDIR)/reshape 7 32 32 1 p1.ra rp1.ra					;\
	$(TOOLDIR)/reshape 7 32 32 1 p2.ra rp2.ra					;\
	$(TOOLDIR)/join 15 rp1.ra rp2.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TST_REF_KSP): phantom join scale
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r5 -k kphan1.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r6 -k kphan2.ra				;\
	$(TOOLDIR)/join 15 kphan1.ra kphan2.ra kphan.ra					;\
	$(TOOLDIR)/scale 0.00032 kphan.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TST_REF_CIM): $(TST_REF_KSP) fft
	$(TOOLDIR)/fft -i -u 7 $(TST_REF_KSP) $@

$(TST_REF_IMG): $(TST_REF_CIM) $(TST_COL) fmac
	$(TOOLDIR)/fmac -s8 -C $(TST_REF_CIM) $(TST_COL) $@

$(TST_KSP): fmac $(TST_PAT) $(TST_REF_KSP)
	$(TOOLDIR)/fmac $(TST_REF_KSP) $(TST_PAT) $@

$(TST_COL): phantom normalize repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/repmat 15 2 sens.ra sens.ra						;\
	$(TOOLDIR)/normalize 8 sens.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TST_TRJ): traj
	$(TOOLDIR)/traj -x32 -y32 $@

$(TST_KSP_NC): reshape $(TST_KSP)
	$(TOOLDIR)/reshape 7 1 32 32 $(TST_KSP) $@

$(TST_REF_KSP_NC): reshape $(TST_REF_KSP)
	$(TOOLDIR)/reshape 7 1 32 32 $(TST_REF_KSP) $@

$(TRN_MSK): ones
	set -e										;\
	$(TOOLDIR)/ones 3 1 32 32 $@

tests/test-reconet-nnvn-train: nrmse reconet $(TRN_REF_IMG) $(TRN_KSP) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 							;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --train-algo  e=1 -b2 --valid-data=k=$(TST_KSP),c=$(TST_COL),r=$(TST_REF_IMG) $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --train-algo e=15 -b2 --valid-data=k=$(TST_KSP),c=$(TST_COL),r=$(TST_REF_IMG) $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network varnet --test -a -n $(TST_KSP) $(TST_COL) weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network varnet --test -a -n $(TST_KSP) $(TST_COL) weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 1.5 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train: nrmse reconet $(TRN_REF_IMG) $(TRN_KSP) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 							;\
	$(TOOLDIR)/reconet --network modl --test -n -t --train-algo  e=1 -b2 $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network modl --test -n -t --train-algo e=15 -b2 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network modl --test -a -n $(TST_KSP) $(TST_COL) weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --test -a -n $(TST_KSP) $(TST_COL) weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 1.2 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-ksp: nrmse reconet $(TRN_REF_KSP) $(TRN_KSP) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 ;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp --ksp-training -t --train-algo  e=1 -b2 $(TRN_KSP) $(TRN_COL) weights0  $(TRN_REF_KSP)			;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp --ksp-training -t --train-algo e=5 -b2 $(TRN_KSP) $(TRN_COL) weights01 $(TRN_REF_KSP)			;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp --ksp-training -t -lweights01 --train-algo e=5 -b2 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_KSP)		;\
	$(TOOLDIR)/reconet --network modl --test -a $(TST_KSP) $(TST_COL) weights0 out0.ra										;\
	$(TOOLDIR)/reconet --network modl --test -a $(TST_KSP) $(TST_COL) weights1 out1.ra										;\
	$(TOOLDIR)/nrmse -t 0.25 out0.ra $(TST_REF_IMG) 														;\
	$(TOOLDIR)/nrmse -t 0.2 out1.ra $(TST_REF_IMG)															;\
	echo "Ratio Error: $$( echo "`$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" / `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` | bc -l )"					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 0.9 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then 				 \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"										;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"										;\
		false																			;\
	fi																				;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-noncart: nrmse reconet $(TRN_REF_CIM) $(TRN_KSP_NC) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP_NC) $(TST_COL) $(TRN_TRJ) $(TST_TRJ)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 ;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp -t --train-algo e=1 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights0  $(TRN_REF_CIM)		;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp -t --train-algo e=5 -b2 -I1 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights01 $(TRN_REF_CIM)	;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp -t -lweights01 --train-algo e=5 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights1  $(TRN_REF_CIM)	;\
	$(TOOLDIR)/reconet --network modl --test -a --trajectory=$(TST_TRJ) $(TST_KSP_NC) $(TST_COL) weights0 out0.ra							;\
	$(TOOLDIR)/reconet --network modl --test -a --trajectory=$(TST_TRJ) $(TST_KSP_NC) $(TST_COL) weights1 out1.ra							;\
	$(TOOLDIR)/nrmse -t 0.25 out0.ra $(TST_REF_IMG) 														;\
	$(TOOLDIR)/nrmse -t 0.2 out1.ra $(TST_REF_IMG)															;\
	echo "Ratio Error: $$( echo "`$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" / `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` | bc -l )"					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 0.9 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then 				 \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TST_REWF_IMG)`"											;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TST_REWF_IMG)`"											;\
		false																			;\
	fi																				;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnvarnet-train-cim: nrmse reconet $(TRN_REF_CIM) $(TRN_KSP) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 ;\
	$(TOOLDIR)/reconet --network varnet --test -t --train-algo e=1 -b2 $(TRN_KSP) $(TRN_COL) weights0  $(TRN_REF_CIM)						;\
	$(TOOLDIR)/reconet --network varnet --test -t --train-algo e=5 -b2 $(TRN_KSP) $(TRN_COL) weights1  $(TRN_REF_CIM)						;\
	$(TOOLDIR)/reconet --network varnet --test -a $(TST_KSP) $(TST_COL) weights0 out0.ra										;\
	$(TOOLDIR)/reconet --network varnet --test -a $(TST_KSP) $(TST_COL) weights1 out1.ra										;\
	$(TOOLDIR)/nrmse -t 0.25 out0.ra $(TST_REF_IMG) 														;\
	$(TOOLDIR)/nrmse -t 0.2 out1.ra $(TST_REF_IMG)															;\
	echo "Ratio Error: $$( echo "`$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" / `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` | bc -l )"					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 0.9 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then 				 \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)`"											;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`"											;\
		false																			;\
	fi																				;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-ksp-noncart: nrmse reconet $(TRN_REF_KSP_NC) $(TRN_KSP_NC) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP_NC) $(TST_COL) $(TRN_TRJ) $(TST_TRJ)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 														;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp --ksp-training -t --train-algo  e=1 -b2            --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights0  $(TRN_REF_KSP_NC)	;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp --ksp-training -t --train-algo e=5 -b2             --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights01 $(TRN_REF_KSP_NC)	;\
	$(TOOLDIR)/reconet --network modl --test --no-precomp --ksp-training -t -lweights01 --train-algo e=5 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights1 $(TRN_REF_KSP_NC)	;\
	$(TOOLDIR)/reconet --network modl --test -a --trajectory=$(TST_TRJ) $(TST_KSP_NC) $(TST_COL) weights0 out0.ra										;\
	$(TOOLDIR)/reconet --network modl --test -a --trajectory=$(TST_TRJ) $(TST_KSP_NC) $(TST_COL) weights1 out1.ra										;\
	$(TOOLDIR)/nrmse -t 0.25 out0.ra $(TST_REF_IMG) 														;\
	$(TOOLDIR)/nrmse -t 0.2 out1.ra $(TST_REF_IMG)															;\
	echo "Ratio Error: $$( echo "`$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" / `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` | bc -l )"					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 0.9 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then 				 \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"										;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"										;\
		false																			;\
	fi																				;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-precomp: nrmse reconet $(TRN_KSP) $(TRN_COL) $(TRN_REF_IMG)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 	;\
	$(TOOLDIR)/reconet --network modl 		--test -t --train-algo e=10 -b2 $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)					;\
	$(TOOLDIR)/reconet --network modl --no-precomp 	--test -t --train-algo e=10 -b2 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)					;\
	$(TOOLDIR)/nrmse -t 0.0 weights1 weights0	;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-precomp-noncart: nrmse reconet $(TRN_KSP_NC) $(TRN_COL) $(TRN_REF_IMG) $(TRN_TRJ)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 	;\
	$(TOOLDIR)/reconet --network modl 		--test -t --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights0 $(TRN_REF_IMG)					;\
	$(TOOLDIR)/reconet --network modl --no-precomp 	--test -t --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights1 $(TRN_REF_IMG)					;\
	$(TOOLDIR)/nrmse -t 7.e-6 weights1 weights0	;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-reconet-nnmodl-train-ksp-noncart-init: nrmse reconet $(TRN_REF_KSP_NC) $(TRN_KSP_NC) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP_NC) $(TST_COL) $(TRN_TRJ) $(TST_TRJ)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 ;\
	$(TOOLDIR)/reconet --network modl --test --initial-reco=sense --no-precomp --ksp-training -t --train-algo  e=1 -b2            --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights0  $(TRN_REF_KSP_NC)			;\
	$(TOOLDIR)/reconet --network modl --test --initial-reco=sense --no-precomp --ksp-training -t --train-algo e=5 -b2             --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights01 $(TRN_REF_KSP_NC)			;\
	$(TOOLDIR)/reconet --network modl --test --initial-reco=sense --no-precomp --ksp-training -t -lweights01 --train-algo e=5 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights1 $(TRN_REF_KSP_NC)		;\
	$(TOOLDIR)/reconet --network modl --test --initial-reco=sense -a --trajectory=$(TST_TRJ) $(TST_KSP_NC) $(TST_COL) weights0 out0.ra										;\
	$(TOOLDIR)/reconet --network modl --test --initial-reco=sense -a --trajectory=$(TST_TRJ) $(TST_KSP_NC) $(TST_COL) weights1 out1.ra										;\
	$(TOOLDIR)/nrmse -t 0.25 out0.ra $(TST_REF_IMG) 														;\
	$(TOOLDIR)/nrmse -t 0.2 out1.ra $(TST_REF_IMG)															;\
	echo "Ratio Error: $$( echo "`$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" / `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` | bc -l )"					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 0.9 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then 				 \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"										;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"										;\
		false																			;\
	fi																				;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-basis: nrmse $(TESTS_OUT)/weights.ra reconet repmat ones scale resize	\
	$(TRN_KSP_NC) $(TRN_REF_IMG) $(TRN_COL) $(TST_KSP_NC) $(TST_REF_IMG) $(TST_COL) $(TRN_KSP) $(TST_KSP) $(WGH)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2				;\
	$(TOOLDIR)/repmat 6 2 $(TRN_REF_IMG) trn_ref.ra					;\
	$(TOOLDIR)/repmat 6 2 $(TST_REF_IMG) tst_ref.ra					;\
	$(TOOLDIR)/resize 5 2 $(TRN_KSP_NC)  trn_ksp.ra				;\
	$(TOOLDIR)/resize 5 2 $(TST_KSP_NC)  tst_ksp.ra				;\
	$(TOOLDIR)/resize 5 2 $(WGH) pat.ra						;\
	$(TOOLDIR)/ones 7 1 1 1 1 1 1 2 bas1								;\
	$(TOOLDIR)/scale 0.5 bas1 bas2							;\
	$(TOOLDIR)/resize 5 2 bas2 bas									;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -t -n --train-algo  e=1 -b2 --trajectory=$(TRN_TRJ) trn_ksp.ra $(TRN_COL) weights0 trn_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -t -n --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) trn_ksp.ra $(TRN_COL) weights1 trn_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -a -n --trajectory=$(TST_TRJ) tst_ksp.ra $(TST_COL) weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -a -n --trajectory=$(TST_TRJ) tst_ksp.ra $(TST_COL) weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra tst_ref.ra` <= 1.3 * `$(TOOLDIR)/nrmse out1.ra tst_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra tst_ref.ra`"				;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra tst_ref.ra`"				;\
		false											;\
	fi												;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-basis-precomp: nrmse $(TESTS_OUT)/weights.ra reconet repmat ones scale resize	\
	$(TRN_KSP_NC) $(TRN_REF_IMG) $(TRN_COL) $(TST_KSP_NC) $(TST_REF_IMG) $(TST_COL) $(TRN_KSP) $(TST_KSP) $(WGH)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2				;\
	$(TOOLDIR)/repmat 6 2 $(TRN_REF_IMG) trn_ref.ra					;\
	$(TOOLDIR)/repmat 6 2 $(TST_REF_IMG) tst_ref.ra					;\
	$(TOOLDIR)/resize 5 2 $(TRN_KSP_NC)  trn_ksp.ra				;\
	$(TOOLDIR)/resize 5 2 $(TST_KSP_NC)  tst_ksp.ra				;\
	$(TOOLDIR)/resize 5 2 $(WGH) pat.ra						;\
	$(TOOLDIR)/ones 7 1 1 1 1 1 1 2 bas1								;\
	$(TOOLDIR)/scale 0.5 bas1 bas2							;\
	$(TOOLDIR)/resize 5 2 bas2 bas									;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -t --no-precomp --train-algo  e=1 -b2 --trajectory=$(TRN_TRJ) trn_ksp.ra $(TRN_COL) weights0 trn_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -t --no-precomp --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) trn_ksp.ra $(TRN_COL) weights1 trn_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -a --trajectory=$(TST_TRJ) tst_ksp.ra $(TST_COL) weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -a --trajectory=$(TST_TRJ) tst_ksp.ra $(TST_COL) weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra tst_ref.ra` <= 1.5 * `$(TOOLDIR)/nrmse out1.ra tst_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra tst_ref.ra`"				;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra tst_ref.ra`"				;\
		false											;\
	fi												;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnvn-train-max-eigen: nrmse reconet $(TRN_KSP) $(TRN_COL) $(TRN_REF_IMG) $(TST_KSP) $(TST_COL) $(TST_REF_IMG)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --data-consistency=gradient-max-eigen --train-algo e=1  -b2 $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)		;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --data-consistency=gradient-max-eigen --train-algo e=20 -b2 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)		;\
	$(TOOLDIR)/reconet --network varnet --test -a -n --data-consistency=gradient-max-eigen $(TST_KSP) $(TST_COL) weights0 out0.ra						;\
	$(TOOLDIR)/reconet --network varnet --test -a -n --data-consistency=gradient-max-eigen $(TST_KSP) $(TST_COL) weights1 out1.ra						;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 1.27 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then 					 \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"											;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"											;\
		false																				;\
	fi																					;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-reconet-nnvn-train-gpu: nrmse reconet $(TRN_REF_IMG) $(TRN_KSP) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 							;\
	$(TOOLDIR)/reconet -g --network varnet --test -n -t --train-algo  e=1 -b2 --valid-data=k=$(TST_KSP),c=$(TST_COL),r=$(TST_REF_IMG) $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet -g --network varnet --test -n -t --train-algo e=15 -b2 --valid-data=k=$(TST_KSP),c=$(TST_COL),r=$(TST_REF_IMG) $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet -g --network varnet --test -a -n $(TST_KSP) $(TST_COL) weights0 out0.ra					;\
	$(TOOLDIR)/reconet -g --network varnet --test -a -n $(TST_KSP) $(TST_COL) weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 1.5 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-gpu: nrmse reconet $(TRN_REF_IMG) $(TRN_KSP) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 							;\
	$(TOOLDIR)/reconet -g --network modl --test -n -t --train-algo  e=1 -b2 $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet -g --network modl --test -n -t --train-algo e=20 -b2 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet -g --network modl --test -a -n $(TST_KSP) $(TST_COL) weights0 out0.ra					;\
	$(TOOLDIR)/reconet -g --network modl --test -a -n $(TST_KSP) $(TST_COL) weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 1.3 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-ksp-gpu: nrmse reconet $(TRN_REF_KSP) $(TRN_KSP) $(TRN_COL) $(TST_REF_IMG) $(TST_KSP) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 ;\
	$(TOOLDIR)/reconet -g --network modl --test --no-precomp --ksp-training -t --train-algo  e=1 -b2 $(TRN_KSP) $(TRN_COL) weights0  $(TRN_REF_KSP)			;\
	$(TOOLDIR)/reconet -g --network modl --test --no-precomp --ksp-training -t --train-algo e=5 -b2 $(TRN_KSP) $(TRN_COL) weights01 $(TRN_REF_KSP)			;\
	$(TOOLDIR)/reconet -g --network modl --test --no-precomp --ksp-training -t -lweights01 --train-algo e=5 -b2 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_KSP)		;\
	$(TOOLDIR)/reconet -g --network modl --test -a $(TST_KSP) $(TST_COL) weights0 out0.ra										;\
	$(TOOLDIR)/reconet -g --network modl --test -a $(TST_KSP) $(TST_COL) weights1 out1.ra										;\
	$(TOOLDIR)/nrmse -t 0.25 out0.ra $(TST_REF_IMG) 														;\
	$(TOOLDIR)/nrmse -t 0.2 out1.ra $(TST_REF_IMG)															;\
	echo "Ratio Error: $$( echo "`$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" / `$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` | bc -l )"					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TST_REF_IMG)` <= 0.9 * `$(TOOLDIR)/nrmse out1.ra $(TST_REF_IMG)`" | bc ) ] ; then 				 \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"										;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"										;\
		false																			;\
	fi																				;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-mpi: bart nrmse reconet $(TRN_KSP) $(TRN_COL) $(TRN_REF_IMG)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 	;\
	                 $(TOOLDIR)/reconet --network modl --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b4 $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	mpirun -n 2 $(ROOTDIR)/bart reconet --network modl --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b4 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/nrmse -t 1.e-4 weights1 weights0	;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-mpi-noncart: bart nrmse reconet $(TRN_KSP_NC) $(TRN_COL) $(TRN_REF_IMG) $(TRN_TRJ)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 	;\
	                 $(TOOLDIR)/reconet --network modl --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	mpirun -n 2 $(ROOTDIR)/bart reconet --network modl --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/nrmse -t 1.e-6 weights1 weights0	;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-mpi-gpu: bart nrmse reconet $(TRN_KSP) $(TRN_COL) $(TRN_REF_IMG)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 	;\
	                 $(TOOLDIR)/reconet --network modl -g --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b4 $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	mpirun -n 2 $(ROOTDIR)/bart reconet --network modl -g --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b4 $(TRN_KSP) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/nrmse -t 1.e-4 weights1 weights0	;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-mpi-gpu-noncart: bart nrmse reconet $(TRN_KSP_NC) $(TRN_COL) $(TRN_REF_IMG) $(TRN_TRJ)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	        			    $(TOOLDIR)/reconet --network modl -g --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights0 $(TRN_REF_IMG)	;\
	BART_GPU_STREAMS=2 mpirun -n 2 $(ROOTDIR)/bart reconet --network modl -g --resnet-block=no-batch-normalization --test -t --train-algo e=10 -b2 --trajectory=$(TRN_TRJ) $(TRN_KSP_NC) $(TRN_COL) weights1 $(TRN_REF_IMG)	;\
	$(TOOLDIR)/nrmse -t 1.e-3 weights1 weights0	;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow2: nrmse multicfl reconet $(TRN_KSP) $(TRN_REF_IMG) $(TRN_COL) $(TST_KSP) $(TST_REF_IMG) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(ROOTDIR)/python python3 $(ROOTDIR)/tests/network_tf2.py ;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1            $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)		;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_b $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_t $(TRN_REF_IMG)	;\
	$(TOOLDIR)/multicfl -s weights_b b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf2_resnet; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow1: nrmse multicfl reconet $(TRN_KSP) $(TRN_REF_IMG) $(TRN_COL) $(TST_KSP) $(TST_REF_IMG) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(ROOTDIR)/python python3 $(ROOTDIR)/tests/network_tf1.py ;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1            $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)		;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_b $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf1_resnet  -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_t $(TRN_REF_IMG)	;\
	$(TOOLDIR)/multicfl -s weights_b b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf1_resnet.*; rm *.hdr ; rm *.cfl; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow1-gpu: nrmse multicfl reconet $(TRN_KSP) $(TRN_REF_IMG) $(TRN_COL) $(TST_KSP) $(TST_REF_IMG) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(ROOTDIR)/python python3 $(ROOTDIR)/tests/network_tf1.py ;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1            $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)		;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf1_resnet    -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_t $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf1_resnet -g -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_g $(TRN_REF_IMG)	;\
	$(TOOLDIR)/multicfl -s weights_g b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf1_resnet.*; rm *.hdr ; rm *.cfl; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow2-gpu: nrmse multicfl reconet $(TRN_KSP) $(TRN_REF_IMG) $(TRN_COL) $(TST_KSP) $(TST_REF_IMG) $(TST_COL)
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(ROOTDIR)/python python3 $(ROOTDIR)/tests/network_tf2.py ;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1            $(TRN_KSP) $(TRN_COL) weights0 $(TRN_REF_IMG)		;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/    -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_t $(TRN_REF_IMG)	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -g -t -n --train-algo e=10 -b2 -I3 -lweights0 $(TRN_KSP) $(TRN_COL) weights_g $(TRN_REF_IMG)	;\
	$(TOOLDIR)/multicfl -s weights_g b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf1_resnet.*; rm *.hdr ; rm *.cfl; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-reconet-nnvn-train
TESTS += tests/test-reconet-nnmodl-train
TESTS += tests/test-reconet-nnmodl-train-ksp
TESTS += tests/test-reconet-nnmodl-train-noncart
TESTS += tests/test-reconet-nnvarnet-train-cim
TESTS += tests/test-reconet-nnmodl-train-ksp-noncart
TESTS += tests/test-reconet-nnmodl-train-precomp
TESTS += tests/test-reconet-nnmodl-train-precomp-noncart
TESTS += tests/test-reconet-nnmodl-train-ksp-noncart-init
TESTS += tests/test-reconet-nnmodl-train-basis

TESTS_MPI += tests/test-reconet-nnmodl-train-mpi

TESTS += tests/test-reconet-nnmodl-train-basis-precomp
TESTS += tests/test-reconet-nnvn-train-max-eigen

TESTS_GPU += tests/test-reconet-nnvn-train-gpu
TESTS_GPU += tests/test-reconet-nnmodl-train-gpu
TESTS_GPU += tests/test-reconet-nnmodl-train-ksp-gpu

ifeq ($(TENSORFLOW),1)
TESTS_MPI += tests/test-reconet-nnmodl-train-mpi
TESTS_MPI += tests/test-reconet-nnmodl-train-mpi-noncart
endif

ifeq ($(TENSORFLOW),1)
TESTS += tests/test-reconet-nnmodl-tensorflow2
TESTS += tests/test-reconet-nnmodl-tensorflow1

TESTS_gpu += tests/test-reconet-nnmodl-tensorflow1-gpu
TESTS_gpu += tests/test-reconet-nnmodl-tensorflow2-gpu
endif

ifeq ($(MPI), 1)
TESTS_GPU += tests/test-reconet-nnmodl-train-mpi-gpu-noncart tests/test-reconet-nnmodl-train-mpi-gpu
endif


# for math in lowmem test
SHELL := /bin/bash

$(TESTS_OUT)/pattern.ra: poisson reshape
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e $(TESTS_TMP)/poisson.ra	;\
	$(TOOLDIR)/reshape 7 32 32 1 poisson.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/weights.ra: poisson reshape
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/poisson -Y 32 -y 2.5 -Z 32 -z 1.5 -C 8 -e $(TESTS_TMP)/poisson.ra	;\
	$(TOOLDIR)/reshape 15 1 32 32 1 poisson.ra $@					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)


$(TESTS_OUT)/pattern_batch.ra: poisson reshape join
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

$(TESTS_OUT)/train_kspace.ra: phantom join scale fmac $(TESTS_OUT)/pattern.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r1 -k kp1.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r2 -k kp2.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r3 -k kp3.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r4 -k kp4.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r5 -k kp5.ra					;\
	$(TOOLDIR)/join 15 kp1.ra kp2.ra kp3.ra kp4.ra kp5.ra kphan.ra			;\
	$(TOOLDIR)/scale 32 kphan.ra kphan.ra						;\
	$(TOOLDIR)/fmac kphan.ra $(TESTS_OUT)/pattern.ra $@				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)						;\

$(TESTS_OUT)/train_kspace_batch_pattern.ra: phantom join scale fmac $(TESTS_OUT)/pattern_batch.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r1 -k kp1.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r2 -k kp2.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r3 -k kp3.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r4 -k kp4.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r5 -k kp5.ra					;\
	$(TOOLDIR)/join 15 kp1.ra kp2.ra kp3.ra kp4.ra kp5.ra kphan.ra			;\
	$(TOOLDIR)/scale 32 kphan.ra kphan.ra						;\
	$(TOOLDIR)/fmac kphan.ra $(TESTS_OUT)/pattern_batch.ra $@			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/train_ref.ra: phantom join scale rss fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r1 phan1.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r2 phan2.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r3 phan3.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r4 phan4.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r5 phan5.ra					;\
	$(TOOLDIR)/join 15 phan1.ra phan2.ra phan3.ra phan4.ra phan5.ra phan.ra		;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/fmac phan.ra scale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)						;\

$(TESTS_OUT)/train_ref_ksp.ra: phantom join scale rss fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r1 -k phan1.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r2 -k phan2.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r3 -k phan3.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r4 -k phan4.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r5 -k phan5.ra					;\
	$(TOOLDIR)/join 15 phan1.ra phan2.ra phan3.ra phan4.ra phan5.ra phan.ra		;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/fmac phan.ra scale.ra kphan.ra					;\
	$(TOOLDIR)/scale 32 kphan.ra $@							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)	

$(TESTS_OUT)/train_sens.ra: phantom rss invert fmac repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/repmat 15 5 sens.ra sens.ra						;\
	$(TOOLDIR)/invert scale.ra iscale.ra						;\
	$(TOOLDIR)/fmac sens.ra iscale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/test_kspace.ra: phantom join scale fmac $(TESTS_OUT)/pattern.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r5 -k kphan5.ra				;\
	$(TOOLDIR)/phantom -x 32 -N5 -s4 -r6 -k kphan6.ra				;\
	$(TOOLDIR)/join 15 kphan5.ra kphan6.ra kphan.ra					;\
	$(TOOLDIR)/scale 32 kphan.ra kphan.ra						;\
	$(TOOLDIR)/fmac kphan.ra $(TESTS_OUT)/pattern.ra $@				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/test_ref.ra: phantom join scale rss fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r5 phan5.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r6 phan6.ra					;\
	$(TOOLDIR)/join 15 phan5.ra phan6.ra phan.ra					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/fmac phan.ra scale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/test_ref_ksp.ra: phantom join scale rss fmac
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r5 -k phan5.ra					;\
	$(TOOLDIR)/phantom -x 32 -N5 -r6 -k phan6.ra					;\
	$(TOOLDIR)/join 15 phan5.ra phan6.ra phan.ra					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/fmac phan.ra scale.ra kphan.ra					;\
	$(TOOLDIR)/scale 32 kphan.ra $@							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/test_sens.ra: phantom rss invert fmac repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -x32 -S4 sens.ra						;\
	$(TOOLDIR)/rss 8 sens.ra scale.ra						;\
	$(TOOLDIR)/repmat 15 2 sens.ra sens.ra						;\
	$(TOOLDIR)/invert scale.ra iscale.ra						;\
	$(TOOLDIR)/fmac sens.ra iscale.ra $@						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)

$(TESTS_OUT)/traj_net.ra: traj
	set -e 										;\
	$(TOOLDIR)/traj -x32 -y32 $@

$(TESTS_OUT)/test_kspace_noncart.ra: reshape $(TESTS_OUT)/test_kspace.ra
	set -e										;\
	$(TOOLDIR)/reshape 7 1 32 32 $(TESTS_OUT)/test_kspace.ra $@

$(TESTS_OUT)/train_kspace_noncart.ra: reshape $(TESTS_OUT)/train_kspace.ra
	set -e										;\
	$(TOOLDIR)/reshape 7 1 32 32 $(TESTS_OUT)/train_kspace.ra $@

$(TESTS_OUT)/train_ref_ksp_noncart.ra: reshape $(TESTS_OUT)/train_ref_ksp.ra
	set -e										;\
	$(TOOLDIR)/reshape 7 1 32 32 $(TESTS_OUT)/train_ref_ksp.ra $@

tests/test-reconet-nnvn-train: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --train-algo e=20 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network varnet --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network varnet --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.27 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=10 -b2 -I1 --valid-data=pattern=$(TESTS_OUT)/pattern.ra,kspace=$(TESTS_OUT)/test_kspace.ra,coil=$(TESTS_OUT)/test_sens.ra,ref=$(TESTS_OUT)/test_ref.ra --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --test -lweights01 -n -t --train-algo e=10 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-ksp: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref_ksp.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref_ksp.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=1 -b2 --ksp-training --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref_ksp.ra		;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=30 -b2 -I1 --ksp-training --valid-data=pattern=$(TESTS_OUT)/pattern.ra,kspace=$(TESTS_OUT)/test_kspace.ra,coil=$(TESTS_OUT)/test_sens.ra,ref=$(TESTS_OUT)/test_ref_ksp.ra --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref_ksp.ra	;\
	$(TOOLDIR)/reconet --network modl --test -lweights01 -n -t --train-algo e=30 -b2 --ksp-training --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref_ksp.ra	;\
	$(TOOLDIR)/reconet --network modl --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref_ksp.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref_ksp.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref_ksp.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref_ksp.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-noncart: nrmse $(TESTS_OUT)/weights.ra reconet \
	$(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra \
	$(TESTS_OUT)/traj_net.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2												;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/weights.ra --trajectory=$(TESTS_OUT)/traj_net.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=10 -b2 -I1 --valid-data=trajectory=$(TESTS_OUT)/traj_net.ra,pattern=$(TESTS_OUT)/weights.ra,kspace=$(TESTS_OUT)/test_kspace_noncart.ra,coil=$(TESTS_OUT)/test_sens.ra,ref=$(TESTS_OUT)/test_ref.ra --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --test -lweights01 -n -t --train-algo e=10 -b2 --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-noncart-ksp: nrmse $(TESTS_OUT)/weights.ra reconet \
	$(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_ref_ksp_noncart.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_ref_ksp.ra $(TESTS_OUT)/test_sens.ra \
	$(TESTS_OUT)/traj_net.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2	;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=1 -b2 --ksp-training --pattern=$(TESTS_OUT)/weights.ra --trajectory=$(TESTS_OUT)/traj_net.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref_ksp_noncart.ra ;\
	$(TOOLDIR)/reconet --network modl --test -t -n --train-algo e=40 -b2 -I1 --ksp-training --valid-data=trajectory=$(TESTS_OUT)/traj_net.ra,pattern=$(TESTS_OUT)/weights.ra,kspace=$(TESTS_OUT)/test_kspace_noncart.ra,coil=$(TESTS_OUT)/test_sens.ra,ref=$(TESTS_OUT)/test_ref_ksp_noncart.ra --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref_ksp_noncart.ra	;\
	$(TOOLDIR)/reconet --network modl --test -lweights01 -n -t --train-algo e=40 -b2 --ksp-training --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref_ksp_noncart.ra	;\
	$(TOOLDIR)/reconet --network modl --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra	;\
	$(TOOLDIR)/reconet --network modl --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra	;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref_ksp.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref_ksp.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref_ksp.ra`"	;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref_ksp.ra`"	;\
		false	;\
	fi ;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-noncart-init: nrmse $(TESTS_OUT)/weights.ra reconet			\
	$(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra	\
	$(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra		\
	$(TESTS_OUT)/traj_net.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2												;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense --test -t -n --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/weights.ra --trajectory=$(TESTS_OUT)/traj_net.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense --test -t -n --train-algo e=10 -b2 --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.04 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnvn-train-noncart-init: nrmse $(TESTS_OUT)/weights.ra reconet			\
	$(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra	\
	$(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra		\
	$(TESTS_OUT)/traj_net.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2												;\
	$(TOOLDIR)/reconet --network varnet --initial-reco=sense,fix-lambda=0.1 --test -t -n --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/weights.ra --trajectory=$(TESTS_OUT)/traj_net.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network varnet --initial-reco=sense,fix-lambda=0.1 --test -t -n --train-algo e=10 -b2 --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network varnet --initial-reco=sense,fix-lambda=0.1 --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network varnet --initial-reco=sense,fix-lambda=0.1 --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=$(TESTS_OUT)/weights.ra $(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-basis: nrmse $(TESTS_OUT)/weights.ra reconet repmat ones scale resize	\
	$(TESTS_OUT)/train_kspace_noncart.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra	\
	$(TESTS_OUT)/test_kspace_noncart.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra		\
	$(TESTS_OUT)/traj_net.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2				;\
	$(TOOLDIR)/repmat 6 2 $(TESTS_OUT)/train_ref.ra trn_ref.ra					;\
	$(TOOLDIR)/repmat 6 2 $(TESTS_OUT)/test_ref.ra tst_ref.ra					;\
	$(TOOLDIR)/resize 5 2 $(TESTS_OUT)/train_kspace_noncart.ra trn_ksp.ra				;\
	$(TOOLDIR)/resize 5 2 $(TESTS_OUT)/test_kspace_noncart.ra tst_ksp.ra				;\
	$(TOOLDIR)/resize 5 2 $(TESTS_OUT)/weights.ra pat.ra						;\
	$(TOOLDIR)/ones 7 1 1 1 1 1 1 2 bas1								;\
	$(TOOLDIR)/scale 0.70710678118 bas1 bas2							;\
	$(TOOLDIR)/resize 5 2 bas2 bas									;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -t -n --train-algo e=1 -b2 --pattern=pat.ra --trajectory=$(TESTS_OUT)/traj_net.ra trn_ksp.ra $(TESTS_OUT)/train_sens.ra weights0 trn_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -t -n --train-algo e=20 -b2 --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=pat.ra trn_ksp.ra $(TESTS_OUT)/train_sens.ra weights1 trn_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=pat.ra tst_ksp.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --initial-reco=sense -Bbas --test -a -n --trajectory=$(TESTS_OUT)/traj_net.ra --pattern=pat.ra tst_ksp.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra tst_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra tst_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra tst_ref.ra`"				;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra tst_ref.ra`"				;\
		false											;\
	fi												;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnvn-train-max-eigen: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --data-consistency=gradient-max-eigen --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network varnet --test -n -t --data-consistency=gradient-max-eigen --train-algo e=20 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network varnet --test -a -n --data-consistency=gradient-max-eigen --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network varnet --test -a -n --data-consistency=gradient-max-eigen --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.27 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnunet-train: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet --network unet --test -t -n --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network unet --test -t -n --train-algo e=10 -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network unet --test -lweights01 -n -t --train-algo e=10 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network unet --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network unet --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnvn-train-gpu: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet -g --network varnet --test -n -t --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet -g --network varnet --test -n -t --train-algo e=20 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet -g --network varnet --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet -g --network varnet --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.28 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnvn-train-mpi: bart $(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 											;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/pattern.ra pattern													;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/train_kspace.ra train_kspace												;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/train_sens.ra train_sens												;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/train_ref.ra train_ref												;\
	mpirun -n 2 $(ROOTDIR)/bart reconet --network varnet --test -n -t --train-algo e=5 -b4 --pattern=pattern train_kspace train_sens weights2 train_ref	;\
	$(ROOTDIR)/bart reconet --network varnet --test -n -t --train-algo e=5 -b4 --pattern=pattern train_kspace train_sens weights1 train_ref			;\
	$(ROOTDIR)/bart nrmse -t 0.000001 weights1 weights2													;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-mpi: bart $(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 											;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/pattern.ra pattern													;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/train_kspace.ra train_kspace												;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/train_sens.ra train_sens												;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/train_ref.ra train_ref												;\
	$(ROOTDIR)/bart reconet --network modl --test -n -t --train-algo e=5 -b4 --pattern=pattern train_kspace train_sens weights1 train_ref			;\
	mpirun -n 2 $(ROOTDIR)/bart reconet --network modl --test -n -t --train-algo e=5 -b4 --pattern=pattern train_kspace train_sens weights2 train_ref	;\
	$(ROOTDIR)/bart nrmse -t 0.00002 weights1 weights2													;\
	rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-train-gpu: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	$(TOOLDIR)/reconet -g --network modl --test -t -n --train-algo e=1 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet -g --network modl --test -t -n --train-algo e=10 -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet -g --network modl --test -lweights01 -n -t --train-algo e=10 -b2 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet -g --network modl --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet -g --network modl --test -a -n --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow2: nrmse multicfl $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_b $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_t $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/multicfl -s weights_b b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf2_resnet; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow1: nrmse multicfl $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf1.py ;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_b $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf1_resnet -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_t $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/multicfl -s weights_b b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf1_resnet.*; rm *.hdr ; rm *.cfl; rm *.map ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-reconet-nnmodl-train-tensorflow2: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -t -n --train-algo e=1 -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -t -n --train-algo e=10 -b2 -I1 --valid-data=pattern=$(TESTS_OUT)/pattern.ra,kspace=$(TESTS_OUT)/test_kspace.ra,coil=$(TESTS_OUT)/test_sens.ra,ref=$(TESTS_OUT)/test_ref.ra --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -n -t --train-algo e=10 -b2 -I3 -lweights01 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -a -n -I3 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl --tensorflow p=tf2_resnet/ -a -n -I3 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm -r tf2_resnet; rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow2-gpu: nrmse multicfl $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
	$(TOOLDIR)/reconet --network modl -g --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl -g --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_b $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl -g --tensorflow p=tf2_resnet/ -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_t $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/multicfl -s weights_b b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf2_resnet; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-reconet-nnmodl-tensorflow1-gpu: nrmse multicfl $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf1.py ;\
	$(TOOLDIR)/reconet --network modl -g --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=1  -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl -g --resnet-block L=3,F=8,no-batch-normalization,no-bias -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_b $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl -g --tensorflow p=tf1_resnet -t -n --train-algo e=10 -b2 -I3 -lweights0 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights_t $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/multicfl -s weights_b b1 b2 b3 b4		;\
	$(TOOLDIR)/multicfl -s weights_t t1 t2 t3 t4		;\
	$(TOOLDIR)/nrmse -t0.0005 b1 t1					;\
	$(TOOLDIR)/nrmse -t0.0005 b2 t2					;\
	$(TOOLDIR)/nrmse -t0.0050 b3 t3					;\
	$(TOOLDIR)/nrmse -t0.0005 b4 t4					;\
	rm -r tf1_resnet.pb; rm tf1_resnet.map; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-reconet-nnmodl-train-tensorflow2-gpu: nrmse $(TESTS_OUT)/pattern.ra reconet \
	$(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_ref.ra $(TESTS_OUT)/train_sens.ra \
	$(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_ref.ra $(TESTS_OUT)/test_sens.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=2 													;\
	CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=$(TOOLDIR)/python python $(TOOLDIR)/tests/network_tf2.py ;\
	$(TOOLDIR)/reconet --network modl -g --tensorflow p=tf2_resnet/ -t -n --train-algo e=1 -b2 -I1 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights0 $(TESTS_OUT)/train_ref.ra		;\
	$(TOOLDIR)/reconet --network modl -g --tensorflow p=tf2_resnet/ -t -n --train-algo e=10 -b2 -I1 --valid-data=pattern=$(TESTS_OUT)/pattern.ra,kspace=$(TESTS_OUT)/test_kspace.ra,coil=$(TESTS_OUT)/test_sens.ra,ref=$(TESTS_OUT)/test_ref.ra --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights01 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl -g --tensorflow p=tf2_resnet/ -n -t --train-algo e=10 -b2 -I3 -lweights01 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/train_kspace.ra $(TESTS_OUT)/train_sens.ra weights1 $(TESTS_OUT)/train_ref.ra	;\
	$(TOOLDIR)/reconet --network modl -g --tensorflow p=tf2_resnet/ -a -n -I3 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights0 out0.ra					;\
	$(TOOLDIR)/reconet --network modl -g --tensorflow p=tf2_resnet/ -a -n -I3 --pattern=$(TESTS_OUT)/pattern.ra $(TESTS_OUT)/test_kspace.ra $(TESTS_OUT)/test_sens.ra weights1 out1.ra					;\
	if [ 1 == $$( echo "`$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra` <= 1.05 * `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`" | bc ) ] ; then \
		echo "untrained error: `$(TOOLDIR)/nrmse out0.ra $(TESTS_OUT)/test_ref.ra`"		;\
		echo   "trained error: `$(TOOLDIR)/nrmse out1.ra $(TESTS_OUT)/test_ref.ra`"		;\
		false									;\
	fi							;\
	rm -r tf2_resnet; rm *.ra ; rm *.hdr ; rm *.cfl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-reconet-nnvn-train
TESTS += tests/test-reconet-nnvn-train-max-eigen
TESTS += tests/test-reconet-nnmodl-train
TESTS += tests/test-reconet-nnmodl-train-ksp
TESTS += tests/test-reconet-nnmodl-train-noncart
TESTS += tests/test-reconet-nnmodl-train-noncart-ksp
TESTS += tests/test-reconet-nnmodl-train-noncart-init
TESTS += tests/test-reconet-nnvn-train-noncart-init
TESTS += tests/test-reconet-nnmodl-train-basis
ifeq ($(MPI),1)
TESTS_SLOW += tests/test-reconet-nnvn-train-mpi
TESTS_SLOW += tests/test-reconet-nnmodl-train-mpi
endif

TESTS_GPU += tests/test-reconet-nnvn-train-gpu
TESTS_GPU += tests/test-reconet-nnmodl-train-gpu

ifeq ($(TENSORFLOW),1)
TESTS += tests/test-reconet-nnmodl-tensorflow1
TESTS += tests/test-reconet-nnmodl-tensorflow2
TESTS += tests/test-reconet-nnmodl-train-tensorflow2

TESTS_GPU += tests/test-reconet-nnmodl-tensorflow1-gpu
TESTS_GPU += tests/test-reconet-nnmodl-tensorflow2-gpu
TESTS_GPU += tests/test-reconet-nnmodl-train-tensorflow2-gpu
endif



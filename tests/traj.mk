T_TOL=1e-8

TRAJ_TURNS=$(AGUE_REF)/traj/t_turns
O_TRAJ_TURNS=-x128 -y73 -t5 -D

TRAJ_GA_c=$(AGUE_REF)/traj/t_GA_c
O_TRAJ_GA_c=-x128 -y51 -r -G -c

TRAJ_GA_H=$(AGUE_REF)/traj/t_GA_H
O_TRAJ_GA_H=-x128 -y53 -r -H

TRAJ_tiny_GA=$(AGUE_REF)/traj/t_tiny_GA
O_TRAJ_tiny_GA=-x128 -y127 -s11 -G -t10 -D -r

TRAJ_MEMS=$(AGUE_REF)/traj/t_MEMS
O_TRAJ_MEMS=-x128 -y31 -t7 -r -s3 -D -E -e5 -c

TRAJ_MEMS_ASYM=$(AGUE_REF)/traj/t_MEMS_asym
O_TRAJ_MEMS_ASYM=-x128 -d192 -y31 -t7 -r -s3 -D -E -e5 -c

TRAJ_GOLDEN_PARTITIONS=$(AGUE_REF)/traj/t_golden_partitions
O_TRAJ_GOLDEN_PARTITIONS=-x 384 -y 29 -t 1 -m 3 -g -D

tests/test-traj_turns: traj nrmse ${TRAJ_TURNS}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_TURNS} t_turns.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_turns.ra ${TRAJ_TURNS}				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_GA_c: traj nrmse ${TRAJ_GA_c}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_GA_c} t_GA_c.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_GA_c.ra ${TRAJ_GA_c}				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_GA_H: traj nrmse ${TRAJ_GA_H}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_GA_H} t_GA_H.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_GA_H.ra ${TRAJ_GA_H}				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_tiny_GA: traj nrmse ${TRAJ_tiny_GA}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_tiny_GA} t_tiny_GA.ra					;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_tiny_GA.ra ${TRAJ_tiny_GA}			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_MEMS-legacy: traj nrmse ${TRAJ_MEMS}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_MEMS} --mems-legacy t_MEMS.ra				;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_MEMS.ra ${TRAJ_MEMS}				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-traj_MEMS: traj nrmse ${TRAJ_MEMS}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_MEMS} t_MEMS.ra					;\
	$(TOOLDIR)/nrmse -t 5e-7 t_MEMS.ra ${TRAJ_MEMS}					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_MEMS_ASYM-legacy: traj nrmse ${TRAJ_MEMS_ASYM}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_MEMS_ASYM} --mems-legacy t_MEMS_asym.ra		;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_MEMS_asym.ra ${TRAJ_MEMS_ASYM}			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_MEMS_ASYM: traj nrmse ${TRAJ_MEMS_ASYM}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj ${O_TRAJ_MEMS_ASYM} t_MEMS_asym.ra				;\
	$(TOOLDIR)/nrmse -t5e-7 t_MEMS_asym.ra ${TRAJ_MEMS_ASYM}			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-traj_golden_partitions: traj nrmse ${TRAJ_GOLDEN_PARTITIONS}.cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	BART_COMPAT_VERSION="v0.9.00" $(TOOLDIR)/traj ${O_TRAJ_GOLDEN_PARTITIONS} t_golden_partitions.ra	;\
	$(TOOLDIR)/nrmse -t${T_TOL} t_golden_partitions.ra ${TRAJ_GOLDEN_PARTITIONS}				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_AGUE += tests/test-traj_turns tests/test-traj_GA_c tests/test-traj_GA_H tests/test-traj_tiny_GA tests/test-traj_MEMS
TESTS_AGUE += tests/test-traj_MEMS_ASYM tests/test-traj_golden_partitions

tests/test-traj-over: traj scale nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y71 -r traja.ra				;\
	$(TOOLDIR)/traj -x64 -y71 -o2. -r trajb.ra			;\
	$(TOOLDIR)/scale 0.5 traja.ra traja2.ra				;\
	$(TOOLDIR)/nrmse -t 0.0000001 traja2.ra trajb.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-over


tests/test-traj-dccen: traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y71 -r -c traja.ra			;\
	$(TOOLDIR)/traj -x128 -y71 -q-0.5:-0.5:0. -r trajb.ra		;\
	$(TOOLDIR)/nrmse -t 0.0000001 traja.ra trajb.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-dccen



tests/test-traj-dccen-over: traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x64 -y71 -r -c -o2. traja.ra			;\
	$(TOOLDIR)/traj -x64 -y71 -q-0.5:-0.5:0. -r -o2. trajb.ra	;\
	$(TOOLDIR)/nrmse -t 0.0000001 traja.ra trajb.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-dccen-over



# compare customAngle to default angle

tests/test-traj-custom: traj poly nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -x128 -y128 -r traja.ra				;\
	$(TOOLDIR)/poly 128 1 0 0.0245436926 angle.ra			;\
	$(TOOLDIR)/traj -x128 -y128 -r -C angle.ra trajb.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 traja.ra trajb.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-custom


tests/test-traj-rot: traj phantom estshift vec nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -R0. -r -y360 -D t0.ra 				;\
	$(TOOLDIR)/phantom -k -t t0.ra k0.ra 				;\
	$(TOOLDIR)/traj -R30. -r -y360 -D t30.ra			;\
	$(TOOLDIR)/phantom -k -t t30.ra k30.ra 				;\
	$(TOOLDIR)/vec 30. real_shift.ra 				;\
	$(TOOLDIR)/vec `$(TOOLDIR)/estshift 4 k0.ra k30.ra | grep -Eo "[0-9.]+$$"` shift.ra		;\
	$(TOOLDIR)/nrmse -t1e-6 real_shift.ra shift.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rot


tests/test-traj-3D: traj ones scale slice rss nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -3 -x128 -y128 -r traj.ra			;\
	$(TOOLDIR)/ones 3 1 1 128 o.ra					;\
	$(TOOLDIR)/scale 63.5 o.ra a.ra					;\
	$(TOOLDIR)/slice 1 0 traj.ra t.ra				;\
	$(TOOLDIR)/rss 1 t.ra b.ra					;\
	$(TOOLDIR)/nrmse -t 0.0000001 b.ra a.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-3D


tests/test-traj-rational-approx-loop: traj slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 233 -r -A -s 1 --double-base -t 2 traj.ra	;\
	$(TOOLDIR)/slice 10 0 traj.ra o1.ra				;\
	$(TOOLDIR)/slice 10 1 traj.ra o2.ra				;\
	$(TOOLDIR)/nrmse -t 0.00005 o1.ra o2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-loop


tests/test-traj-rational-approx-pattern: traj ones nufft fft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 233 -r -A -s 1 --double-base t.ra		;\
	$(TOOLDIR)/traj -y 233 -r -D t2.ra				;\
	$(TOOLDIR)/ones 3 1 128 233 o.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t.ra o.ra psf.ra		;\
	$(TOOLDIR)/fft 7 psf.ra pattern.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t2.ra o.ra psf2.ra		;\
	$(TOOLDIR)/fft 7 psf2.ra pattern2.ra				;\
	$(TOOLDIR)/nrmse -t 1e-6 pattern.ra pattern2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-pattern


tests/test-traj-rational-approx-pattern2: traj ones nufft fft nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 466 -r -A -s 1 t.ra				;\
	$(TOOLDIR)/traj -y 466 -r -D t2.ra				;\
	$(TOOLDIR)/ones 3 1 128 466 o.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t.ra o.ra psf.ra		;\
	$(TOOLDIR)/fft 7 psf.ra pattern.ra				;\
	$(TOOLDIR)/nufft -a -x128:128:1 t2.ra o.ra psf2.ra		;\
	$(TOOLDIR)/fft 7 psf2.ra pattern2.ra				;\
	$(TOOLDIR)/nrmse -t 3e-6 pattern.ra pattern2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-pattern2


tests/test-traj-double-base: traj slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 2 -r -G -s1 --double-base traj.ra		;\
	$(TOOLDIR)/traj -y 3 -r -G -s1 traj2.ra				;\
	$(TOOLDIR)/slice 2 1 traj.ra o1.ra				;\
	$(TOOLDIR)/slice 2 2 traj2.ra o2.ra				;\
	$(TOOLDIR)/nrmse -t 0. o1.ra o2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-double-base

tests/test-traj-rational-approx-double-base-ga: traj transpose reshape slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 754 -r -D -s 1 t1.ra				;\
	$(TOOLDIR)/reshape 12 2 377 t1.ra t1a.ra			;\
	$(TOOLDIR)/slice 2 0 t1a.ra t1b.ra				;\
	$(TOOLDIR)/transpose 2 3 t1b.ra t1c.ra				;\
	$(TOOLDIR)/traj -y 377 -r -A -s 1 --double-base t2.ra		;\
	$(TOOLDIR)/nrmse -t 0.005 t1c.ra t2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-double-base-ga

tests/test-traj-rational-approx-ga: traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/traj -y 754 -r -D -s 1 t1.ra			;\
	$(TOOLDIR)/traj -y 754 -r -A -s 1 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0.005 t1.ra t2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-ga


tests/test-traj-rational-approx-inc: traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/traj -y 754 -r -A -s 1 -t3 t1.ra				;\
	$(TOOLDIR)/traj -y 754 -r -A --raga-inc 233 -t3 t2.ra			;\
	$(TOOLDIR)/nrmse -t 0. t1.ra t2.ra					;\
	$(TOOLDIR)/traj -y 377 -r -A -s 1 --double-base -t3 t3.ra		;\
	$(TOOLDIR)/traj -y 377 -r -A --raga-inc 233 --double-base -t3 t4.ra	;\
	$(TOOLDIR)/nrmse -t 0. t3.ra t4.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-inc


tests/test-traj-rational-approx-multislice-aligned: traj slice transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 466 -r -D -s 1 -m3 -l traj_gaal.ra		;\
	$(TOOLDIR)/traj -y 466 -r -A -s 1 -m3 -l traj_raga_al.ra	;\
	$(TOOLDIR)/nrmse -t 0.007 traj_gaal.ra traj_raga_al.ra		;\
	$(TOOLDIR)/slice 13 0 traj_raga_al.ra tr0.ra			;\
	$(TOOLDIR)/slice 13 1 traj_raga_al.ra tr1.ra			;\
	$(TOOLDIR)/nrmse -t 0 tr0.ra tr1.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-multislice-aligned

tests/test-traj-rational-approx-multislice: traj transpose slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/traj -y 1 -t 466 -r -D -s 1 -m3 traj_tmp.ra		;\
	$(TOOLDIR)/transpose 2 10 traj_tmp.ra traj_ga.ra		;\
	$(TOOLDIR)/traj      -y 466 -r -A -s 1 -m3 traj_raga.ra		;\
	$(TOOLDIR)/nrmse -t 0.021 traj_ga.ra traj_raga.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-traj-rational-approx-multislice



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


tests/test-pics-eulermaruyama-loop-omp: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(ROOTDIR)/bart ones 6 128 128 1 1 1 4 s.ra						;\
	$(ROOTDIR)/bart zeros 6 128 128 1 1 1 4 z.ra						;\
	$(ROOTDIR)/bart 				      pics --eulermaruyama -S -w1. -s0.01 -i10 -l2 -r1. -p s.ra z.ra s.ra x1.ra	;\
	OMP_NUM_THREADS=4 $(ROOTDIR)/bart -p $$($(ROOTDIR)/bart bitmask 5) -e 4 pics --eulermaruyama -S -w1. -s0.01 -i10 -l2 -r1. -p s.ra z.ra s.ra x2.ra	;\
	$(ROOTDIR)/bart nrmse -t 0.0 x1.ra x2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-eulermaruyama-loop: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(ROOTDIR)/bart ones 6 128 128 1 1 1 4 s.ra						;\
	$(ROOTDIR)/bart zeros 6 128 128 1 1 1 4 z.ra						;\
	$(ROOTDIR)/bart 				      pics --eulermaruyama -S -w1. -s0.01 -i10 -l2 -r1. -p s.ra z.ra s.ra x1.ra	;\
	$(ROOTDIR)/bart -l $$($(ROOTDIR)/bart bitmask 5) -e 4 pics --eulermaruyama -S -w1. -s0.01 -i10 -l2 -r1. -p s.ra z.ra s.ra x2.ra	;\
	$(ROOTDIR)/bart nrmse -t 0.0 x1.ra x2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# This bart nrmse should fail, since the loop dimension is not the last dimension
tests/test-pics-eulermaruyama-loop-fail: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(ROOTDIR)/bart ones 8 128 128 1 1 1 4 1 8 s.ra						;\
	$(ROOTDIR)/bart zeros 8 128 128 1 1 1 4 1 8 z.ra						;\
	$(ROOTDIR)/bart 				      pics --eulermaruyama -S -w1. -s0.01 -i10 -l2 -r1. -p s.ra z.ra s.ra x1.ra	;\
	$(ROOTDIR)/bart -l $$($(ROOTDIR)/bart bitmask 5) -e 4 pics --eulermaruyama -S -w1. -s0.01 -i10 -l2 -r1. -p s.ra z.ra s.ra x2.ra	;\
	! $(ROOTDIR)/bart nrmse -t 1.0 x1.ra x2.ra 2>/dev/null 								;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-pics-cart-loop tests/test-pics-cart-loop_range tests/test-pics-cart-slice tests/test-pics-eulermaruyama-loop tests/test-pics-eulermaruyama-loop-fail

ifeq ($(OMP),1)
TESTS_SLOW += tests/test-pics-cart-loop_range-omp tests/test-pics-eulermaruyama-loop-omp
endif


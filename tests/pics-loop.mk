

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


TESTS += tests/test-pics-cart-loop tests/test-pics-cart-loop_range tests/test-pics-cart-slice

ifeq ($(OMP),1)
TESTS_SLOW += tests/test-pics-cart-loop_range-omp
endif


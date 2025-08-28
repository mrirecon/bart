

# basic 2D FFT

$(TESTS_OUT)/shepplogan_fft.ra: fft $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/fft 7 $(TESTS_OUT)/shepplogan.ra $@


tests/test-fft-basic: scale fft nrmse $(TESTS_OUT)/shepplogan_fft.ra $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/scale 16384 $(TESTS_OUT)/shepplogan.ra shepploganS.ra		;\
	$(TOOLDIR)/fft -i 7 $(TESTS_OUT)/shepplogan_fft.ra shepplogan2.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 shepploganS.ra shepplogan2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


# unitary FFT

$(TESTS_OUT)/shepplogan_fftu.ra: fft $(TESTS_OUT)/shepplogan.ra
	$(TOOLDIR)/fft -u 7 $(TESTS_OUT)/shepplogan.ra $@

tests/test-fft-unitary: fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/fft -u -i 7 $(TESTS_OUT)/shepplogan_fftu.ra shepplogan2u.ra		;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra shepplogan2u.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# uncentered FFT
tests/test-fft-uncentered: fftmod fft nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/fftmod -i 7 $(TESTS_OUT)/shepplogan_fftu.ra shepplogan_fftu2.ra		;\
	$(TOOLDIR)/fft -uni 7 shepplogan_fftu2.ra shepplogan2u.ra			;\
	$(TOOLDIR)/fftmod -i 7 shepplogan2u.ra shepplogan3u.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 $(TESTS_OUT)/shepplogan.ra shepplogan3u.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-fft-shift: resize fft fftshift nrmse $(TESTS_OUT)/shepplogan_fftu.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/resize -c 0 127 1 16 3 5 $(TESTS_OUT)/shepplogan_fftu.ra i.ra	;\
	$(TOOLDIR)/fft 15 i.ra k1.ra			;\
	$(TOOLDIR)/fftshift -b 15 i.ra t1.ra		;\
	$(TOOLDIR)/fft -n 15 t1.ra t2.ra						;\
	$(TOOLDIR)/fftshift 15 t2.ra k2.ra					;\
	$(TOOLDIR)/nrmse -t 1e-6 k1.ra k2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-fft-multi-loop-mpi: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)													;\
	$(ROOTDIR)/bart phantom -B -x 64 bart_phantom.ra 												;\
	$(ROOTDIR)/bart phantom -B -x 64 -k bart_phantom_ksp.ra												;\
	$(ROOTDIR)/bart phantom -G -x 64 geom_phantom.ra 												;\
	$(ROOTDIR)/bart phantom -G -x 64 -k geom_phantom_ksp.ra 											;\
	$(ROOTDIR)/bart phantom -x 64 shepplogan_phantom.ra 												;\
	$(ROOTDIR)/bart phantom -x 64 -k shepplogan_phantom_ksp.ra 											;\
	$(ROOTDIR)/bart fft -iu 7 bart_phantom.ra bart_phantom_fft.ra 											;\
	$(ROOTDIR)/bart fft -iu 7 bart_phantom_ksp.ra bart_phantom_ksp_fft.ra 										;\
	$(ROOTDIR)/bart fft -iu 7 geom_phantom.ra geom_phantom_fft.ra 											;\
	$(ROOTDIR)/bart fft -iu 7 geom_phantom_ksp.ra geom_phantom_ksp_fft.ra 										;\
	$(ROOTDIR)/bart fft -iu 7 shepplogan_phantom.ra shepplogan_phantom_fft.ra 									;\
	$(ROOTDIR)/bart fft -iu 7 shepplogan_phantom_ksp.ra shepplogan_phantom_ksp_fft.ra 								;\
	$(ROOTDIR)/bart join 6 bart_phantom_fft.ra bart_phantom_ksp_fft.ra geom_phantom_fft.ra geom_phantom_ksp_fft.ra phantom_1_fft			;\
	$(ROOTDIR)/bart join 6 geom_phantom_fft.ra geom_phantom_ksp_fft.ra shepplogan_phantom_fft.ra shepplogan_phantom_ksp_fft.ra phantom_2_fft	;\
	$(ROOTDIR)/bart join 6 shepplogan_phantom_ksp_fft.ra bart_phantom_ksp_fft.ra geom_phantom_ksp_fft.ra bart_phantom_ksp_fft.ra phantom_3_fft	;\
	$(ROOTDIR)/bart join 15 phantom_1_fft phantom_2_fft phantom_3_fft phantom_fft_ref 								;\
	$(ROOTDIR)/bart join 6 bart_phantom.ra bart_phantom_ksp.ra geom_phantom.ra geom_phantom_ksp.ra phantom_1 					;\
	$(ROOTDIR)/bart join 6 geom_phantom.ra geom_phantom_ksp.ra shepplogan_phantom.ra shepplogan_phantom_ksp.ra phantom_2				;\
	$(ROOTDIR)/bart join 6 shepplogan_phantom_ksp.ra bart_phantom_ksp.ra geom_phantom_ksp.ra bart_phantom_ksp.ra phantom_3				;\
	$(ROOTDIR)/bart join 15 phantom_1 phantom_2 phantom_3 phantom_stack										;\
	mpirun -n 4 $(ROOTDIR)/bart -p 32832 -e 4:3 fft -iu 7 phantom_stack phantom_stack_fft					;\
	$(ROOTDIR)/bart nrmse -t 1e-5 phantom_fft_ref phantom_stack_fft											;\
	rm *.ra ; rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-fft-multi-loop-omp: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)													;\
	$(ROOTDIR)/bart phantom phan.ra 														;\
	$(ROOTDIR)/bart                fft 2 phan.ra ksp1.ra												;\
	$(ROOTDIR)/bart -p 1 -rphan.ra fft 2 phan.ra ksp2.ra												;\
	$(ROOTDIR)/bart nrmse -t 1e-5 ksp1.ra ksp2.ra													;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-fft-multi-loop-mpi-strided: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)													;\
	$(ROOTDIR)/bart phantom phan.ra 														;\
	$(ROOTDIR)/bart                fft 2 phan.ra ksp1.ra												;\
	mpirun -n 3 $(ROOTDIR)/bart -l 1 -rphan.ra fft 2 phan.ra ksp2.ra								;\
	$(ROOTDIR)/bart nrmse -t 1e-5 ksp1.ra ksp2.ra													;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-fft-basic tests/test-fft-unitary tests/test-fft-uncentered tests/test-fft-shift
TESTS += tests/test-fft-multi-loop-omp

TESTS_MPI += tests/test-fft-multi-loop-mpi tests/test-fft-multi-loop-mpi-strided


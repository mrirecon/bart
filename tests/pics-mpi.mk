

tests/test-pics-cart-batch-mpi: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra img_l2_ref.ra		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p				;\
	mpirun -n 4 $(ROOTDIR)/bart -p 8192 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p		;\
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
	mpirun -n 2 $(ROOTDIR)/bart -l 8192 -s 0 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p03		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_03.ra img_l2_p03										;\
	mpirun -n 4 $(ROOTDIR)/bart -l 8192 -s 1 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p13		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_13.ra img_l2_p13										;\
	mpirun -n 4 $(ROOTDIR)/bart -l 8192 -s 2 -e 3 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p23		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_23.ra img_l2_p23										;\
	mpirun -n 4 $(ROOTDIR)/bart -l 8192 -s 0 -e 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p02		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_02.ra img_l2_p02										;\
	mpirun -n 3 $(ROOTDIR)/bart -l 8192 -s 0 -e 1 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p01		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref_01.ra img_l2_p01										;\
	rm *.ra *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-cart-mpi: bart pics copy nrmse $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/sens_1.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(TOOLDIR)/copy $(TESTS_OUT)/ksp_usamp_1.ra shepplogan_coil_ksp									;\
	$(TOOLDIR)/copy $(TESTS_OUT)/sens_1.ra coils												;\
	mpirun -n 2 $(ROOTDIR)/bart pics --mpi=8 -S -r0.001 shepplogan_coil_ksp coils reco					;\
	                                $(ROOTDIR)/bart pics         -S -r0.001 shepplogan_coil_ksp coils reco_ref				;\
	$(TOOLDIR)/nrmse -t 1e-5 reco_ref reco													;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-cart-mpi_shared: bart pics copy nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/copy $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_coil_ksp				;\
	$(TOOLDIR)/copy $(TESTS_OUT)/coils.ra coils							;\
	mpirun -n 2 $(ROOTDIR)/bart -S pics --mpi=8 -S -r0.001 shepplogan_coil_ksp coils reco		;\
	$(ROOTDIR)/bart pics -S -r0.001 shepplogan_coil_ksp coils reco_ref				;\
	$(TOOLDIR)/nrmse -t 1e-5 reco_ref reco								;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-cart-mpi-sharedcoil: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_1.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra sens_p										;\
	mpirun -n 3 $(ROOTDIR)/bart -S pics --mpi=8 -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p		;\
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
	mpirun -n 2 $(ROOTDIR)/bart -S pics --mpi=$(shell $(ROOTDIR)/bart bitmask 13) -i10 -w 1. -m ksp coils reco		;\
	$(ROOTDIR)/bart pics -i10 -w 1. -m ksp coils ref.ra											;\
	$(ROOTDIR)/bart nrmse -t 1e-5 ref.ra reco 												;\
	rm *.cfl *.hdr *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-cart-mpi-multipledims: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)														;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p					;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p							;\
	mpirun  --host localhost:6 -n 6 $(ROOTDIR)/bart -S pics --mpi=8200 -i10 -S -l2 -r 0.005 ksp_usamp_p sens_p img_l2_p			;\
	$(ROOTDIR)/bart -S pics            -i10 -S -l2 -r 0.005 ksp_usamp_p sens_p img_l2_ref		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref img_l2_p													;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-mpi-multipledims-sharedcoil: bart $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/sens_1.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)														;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p					;\
	$(ROOTDIR)/bart copy $(TESTS_OUT)/sens_1.ra sens_p													;\
	mpirun  --host localhost:6 -n 6 $(ROOTDIR)/bart -S pics --mpi=8200 -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_p			;\
	$(ROOTDIR)/bart -S pics            -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_ref		;\
	$(ROOTDIR)/bart nrmse -t 2e-5 img_l2_ref img_l2_p													;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-noncart-mpi: bart traj phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/traj -r -x128 -o2. -y64 traj.ra								;\
	$(TOOLDIR)/phantom -s4 -t traj.ra ksp									;\
	$(TOOLDIR)/phantom -S4 col										;\
	mpirun -n 2 $(ROOTDIR)/bart -S pics --mpi=8 -S --fista -e -r0.001 -t traj.ra ksp col reco1		;\
	$(ROOTDIR)/bart pics -S --fista -e -r0.001 -t traj.ra ksp col reco2					;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1 reco2									;\
	rm *.cfl *.ra *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# Very slow due to recomposing via mpi 0=> deactivated
tests/test-pics-noncart-mpi-gridH: bart traj phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/traj -r -x128 -y64 -o2 traj									;\
	$(TOOLDIR)/phantom -s4 -t traj ksp									;\
	$(TOOLDIR)/phantom -S4 col										;\
	mpirun -n 2 $(ROOTDIR)/bart -S pics --no-toeplitz --mpi=8 -S --fista -e -r0.001 -t traj ksp col reco1	;\
	$(ROOTDIR)/bart pics --no-toeplitz -S --fista -e -r0.001 -t traj ksp col reco2				;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1 reco2									;\
	rm *.cfl *.ra *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-slice-batch-mpi: bart  $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra $(TESTS_OUT)/img_l2_1.ra $(TESTS_OUT)/img_l2_2.ra $(TESTS_OUT)/img_l2_3.ra $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)											;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/ksp_usamp_2.ra $(TESTS_OUT)/ksp_usamp_3.ra ksp_usamp_p		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/sens_1.ra $(TESTS_OUT)/sens_2.ra $(TESTS_OUT)/sens_3.ra sens_p				;\
	mpirun -n 1 $(ROOTDIR)/bart -l 8192 -s 0 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_0					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_1.ra img_l2_0									;\
	mpirun -n 2 $(ROOTDIR)/bart -l 8192 -s 1 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_1					;\
	$(ROOTDIR)/bart nrmse -t 2e-5 $(TESTS_OUT)/img_l2_2.ra img_l2_1									;\
	mpirun -n 3 $(ROOTDIR)/bart -l 8192 -s 2 pics -S -l2 -r 0.005 -i 3 ksp_usamp_p sens_p img_l2_2					;\
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
	mpirun -n 2 $(ROOTDIR)/bart -p 8192 -e 3 pics -i 3 -ttr_p ksp_rad_p sens1_p pics_rad_p					;\
	$(ROOTDIR)/bart nrmse -t 5e-3 pics_rad_ref pics_rad_p									;\
	rm *.cfl ; rm *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS_MPI += tests/test-pics-cart-batch-mpi tests/test-pics-non-cart-batch-mpi tests/test-pics-cart-slice-batch-mpi tests/test-pics-cart-range-batch-mpi
TESTS_MPI += tests/test-pics-cart-mpi tests/test-pics-noncart-mpi tests/test-pics-mpi-timedim
TESTS_MPI += tests/test-pics-cart-mpi_shared tests/test-pics-cart-mpi-sharedcoil tests/test-pics-cart-mpi-multipledims tests/test-pics-cart-mpi-multipledims-sharedcoil



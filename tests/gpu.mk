

#

tests/test-pics-gpu: pics nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/pics -g -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco1.ra	;\
	$(TOOLDIR)/pics    -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-gpu-noncart: traj scale phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics    -S --fista -e -r0.001 -t traj2.ra ksp.ra o.ra reco1.ra	;\
	$(TOOLDIR)/pics -g -S --fista -e -r0.001 -t traj2.ra ksp.ra o.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-gpu-noncart-gridding: traj scale phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y64 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics                   -S -r0.001 -t traj2.ra ksp.ra o.ra reco1.ra	;\
	$(TOOLDIR)/pics -g --gpu-gridding -S -r0.001 -t traj2.ra ksp.ra o.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.007 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-gpu-weights: pics scale ones nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 2 128 128 weights.ra						;\
	$(TOOLDIR)/scale 0.1 weights.ra weights2.ra					;\
	$(TOOLDIR)/pics    -S -r0.001 -p weights2.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco1.ra	;\
	$(TOOLDIR)/pics -g -S -r0.001 -p weights2.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco2.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 reco2.ra reco1.ra				 	;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# similar to the non-gpu test this had to be relaxed to 0.01
tests/test-pics-gpu-noncart-weights: traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y32 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -s8 -t traj2.ra ksp.ra					;\
	$(TOOLDIR)/ones 4 1 256 32 1 weights.ra						;\
	$(TOOLDIR)/scale 0.1 weights.ra weights2.ra					;\
	$(TOOLDIR)/pics    -S -r0.001 -p weights2.ra -t traj2.ra ksp.ra $(TESTS_OUT)/coils.ra reco1.ra ;\
	$(TOOLDIR)/pics -g -S -r0.001 -p weights2.ra -t traj2.ra ksp.ra $(TESTS_OUT)/coils.ra reco2.ra ;\
	$(TOOLDIR)/nrmse -t 0.010 reco1.ra reco2.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-gpu-llr: traj scale phantom ones pics nrmse $(TESTS_OUT)/shepplogan.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/traj -r -x256 -y48 traj.ra						;\
	$(TOOLDIR)/scale 0.5 traj.ra traj2.ra						;\
	$(TOOLDIR)/phantom -t traj2.ra ksp.ra						;\
	$(TOOLDIR)/ones 3 128 128 1 o.ra						;\
	$(TOOLDIR)/pics -S    -R L:7:7:.02 -u0.1 -e -t traj2.ra ksp.ra o.ra reco_c.ra	;\
	$(TOOLDIR)/pics -S -g -R L:7:7:.02 -u0.1 -e -t traj2.ra ksp.ra o.ra reco_g.ra	;\
	$(TOOLDIR)/nrmse -t 0.0001 reco_c.ra reco_g.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-multigpu: bart copy pics repmat nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/repmat 5 32 $(TESTS_OUT)/shepplogan_coil_ksp.ra kspaces		;\
	$(TOOLDIR)/copy $(TESTS_OUT)/coils.ra coils					;\
	$(ROOTDIR)/bart -l32 -r kspaces pics -g -r0.01 kspaces coils reco1		;\
	$(ROOTDIR)/bart -p32 -r kspaces pics -g -r0.01 kspaces coils reco2		;\
	$(TOOLDIR)/nrmse -t 0.00001 reco1 reco2						;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-gpu-omp: bart nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(ROOTDIR)/bart scale 0.3 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_coil_ksp_s.ra							;\
	$(ROOTDIR)/bart pics    -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco1.ra					;\
	$(ROOTDIR)/bart pics    -S -r0.001 shepplogan_coil_ksp_s.ra $(TESTS_OUT)/coils.ra reco2.ra						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_coil_ksp_s.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/coils.ra $(TESTS_OUT)/coils.ra $(TESTS_OUT)/coils.ra coils						;\
	$(ROOTDIR)/bart join 13 reco1.ra reco2.ra reco1.ra reco_ref.ra										;\
	OMP_NUM_THREADS=2 $(ROOTDIR)/bart -p 8192 -e 3 -t 2 pics -g -S -r0.001 ksp coils reco_p							;\
	$(TOOLDIR)/nrmse -t 0.000001 reco_ref.ra reco_p												;\
	rm *.cfl ; rm *.hdr ; rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-gpu-noncart-weights-omp: bart traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)										;\
	$(ROOTDIR)/bart traj -r -x256 -y32 traj.ra										;\
	$(ROOTDIR)/bart scale 0.5 traj.ra traj1.ra										;\
	$(ROOTDIR)/bart traj -r -R 90 -x256 -y32 traj2.ra									;\
	$(ROOTDIR)/bart scale 0.75 traj2.ra traj2.ra										;\
	$(ROOTDIR)/bart phantom -s8 -t traj.ra ksp.ra										;\
	$(ROOTDIR)/bart phantom -s8 -t traj1.ra ksp1.ra										;\
	$(ROOTDIR)/bart phantom -s8 -t traj2.ra ksp2.ra										;\
	$(ROOTDIR)/bart phantom -S8 coils.ra											;\
	$(ROOTDIR)/bart scale 0.5 coils.ra coils1.ra										;\
	$(ROOTDIR)/bart scale 0.75   coils.ra coils2.ra										;\
	$(ROOTDIR)/bart ones 4 1 256 32 1 weights.ra										;\
	$(ROOTDIR)/bart scale 0.5 weights.ra weights1.ra									;\
	$(ROOTDIR)/bart scale 0.75 weights.ra weights2.ra									;\
	OMP_NUM_THREADS=1 $(ROOTDIR)/bart pics -S -r0.001 -p weights.ra -t traj.ra ksp.ra coils.ra reco1.ra			;\
	OMP_NUM_THREADS=1 $(ROOTDIR)/bart pics -S -r0.001 -p weights1.ra -t traj1.ra ksp1.ra coils1.ra reco2.ra			;\
	OMP_NUM_THREADS=1 $(ROOTDIR)/bart pics -S -r0.001 -p weights2.ra -t traj2.ra ksp2.ra coils2.ra reco3.ra			;\
	$(ROOTDIR)/bart join 13 reco1.ra reco2.ra reco3.ra reco_ref.ra								;\
	$(ROOTDIR)/bart join 13 traj.ra traj1.ra traj2.ra traj_p								;\
	$(ROOTDIR)/bart join 13 ksp.ra ksp1.ra ksp2.ra ksp_p									;\
	$(ROOTDIR)/bart join 13 weights.ra weights1.ra weights2.ra weights_p							;\
	$(ROOTDIR)/bart join 13 coils.ra coils1.ra coils2.ra coils_p								;\
	OMP_NUM_THREADS=2 $(ROOTDIR)/bart -p 8192 -e 3 -t 2 pics -g -S -r0.001 -p weights_p -t traj_p ksp_p coils_p reco_p	;\
	$(TOOLDIR)/nrmse -t 0.01 reco_ref.ra reco_p										;\
	rm *.cfl ; rm *.hdr ; rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-pics-gpu-mpi: bart nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(ROOTDIR)/bart scale 0.3 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_coil_ksp_s.ra							;\
	$(ROOTDIR)/bart pics    -S -r0.001 $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra reco1.ra					;\
	$(ROOTDIR)/bart pics    -S -r0.001 shepplogan_coil_ksp_s.ra $(TESTS_OUT)/coils.ra reco2.ra						;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_coil_ksp_s.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra ksp		;\
	$(ROOTDIR)/bart join 13 $(TESTS_OUT)/coils.ra $(TESTS_OUT)/coils.ra $(TESTS_OUT)/coils.ra coils						;\
	$(ROOTDIR)/bart join 13 reco1.ra reco2.ra reco1.ra reco_ref.ra										;\
	mpirun -n 2 $(ROOTDIR)/bart -p 8192 -e 3 pics -g -S -r0.001 ksp coils reco_p								;\
	$(TOOLDIR)/nrmse -t 0.000001 reco_ref.ra reco_p												;\
	rm *.cfl ; rm *.hdr ; rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-gpu-noncart-weights-mpi: bart traj scale ones phantom pics nrmse $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)									;\
	$(ROOTDIR)/bart traj -r -x256 -y32 traj.ra										;\
	$(ROOTDIR)/bart scale 0.5 traj.ra traj1.ra										;\
	$(ROOTDIR)/bart traj -r -R 90 -x256 -y32 traj2.ra									;\
	$(ROOTDIR)/bart scale 0.75 traj2.ra traj2.ra										;\
	$(ROOTDIR)/bart phantom -s8 -t traj.ra ksp.ra										;\
	$(ROOTDIR)/bart phantom -s8 -t traj1.ra ksp1.ra										;\
	$(ROOTDIR)/bart phantom -s8 -t traj2.ra ksp2.ra										;\
	$(ROOTDIR)/bart phantom -S8 coils.ra											;\
	$(ROOTDIR)/bart scale 0.5 coils.ra coils1.ra										;\
	$(ROOTDIR)/bart scale 0.75   coils.ra coils2.ra										;\
	$(ROOTDIR)/bart ones 4 1 256 32 1 weights.ra										;\
	$(ROOTDIR)/bart scale 0.5 weights.ra weights1.ra									;\
	$(ROOTDIR)/bart scale 0.75 weights.ra weights2.ra									;\
	$(ROOTDIR)/bart pics -S -r0.001 -p weights.ra -t traj.ra ksp.ra coils.ra reco1.ra					;\
	$(ROOTDIR)/bart pics -S -r0.001 -p weights1.ra -t traj1.ra ksp1.ra coils1.ra reco2.ra					;\
	$(ROOTDIR)/bart pics -S -r0.001 -p weights2.ra -t traj2.ra ksp2.ra coils2.ra reco3.ra					;\
	$(ROOTDIR)/bart join 13 reco1.ra reco2.ra reco3.ra reco_ref.ra								;\
	$(ROOTDIR)/bart join 13 traj.ra traj1.ra traj2.ra traj_p								;\
	$(ROOTDIR)/bart join 13 ksp.ra ksp1.ra ksp2.ra ksp_p									;\
	$(ROOTDIR)/bart join 13 weights.ra weights1.ra weights2.ra weights_p							;\
	$(ROOTDIR)/bart join 13 coils.ra coils1.ra coils2.ra coils_p								;\
	mpirun -n 2 $(ROOTDIR)/bart -p 8192 -e 3 pics -g -S -r0.001 -p weights_p -t traj_p ksp_p coils_p reco_p			;\
	$(TOOLDIR)/nrmse -t 0.01 reco_ref.ra reco_p										;\
	rm *.cfl ; rm *.hdr ; rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-pics-cart-mpi-gpu: bart pics copy nrmse $(TESTS_OUT)/shepplogan_coil_ksp.ra $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)										;\
	$(TOOLDIR)/copy $(TESTS_OUT)/shepplogan_coil_ksp.ra shepplogan_coil_ksp							;\
	$(TOOLDIR)/copy $(TESTS_OUT)/coils.ra coils										;\
	mpirun -n 2 $(ROOTDIR)/bart --md-split-mpi-dims=8 pics -g -S -r0.001 shepplogan_coil_ksp coils reco			;\
	            $(ROOTDIR)/bart                       pics -g -S -r0.001 shepplogan_coil_ksp coils reco_ref			;\
	$(TOOLDIR)/nrmse -t 1e-5 reco_ref reco											;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS_GPU += tests/test-pics-gpu tests/test-pics-gpu-noncart tests/test-pics-gpu-noncart-gridding
TESTS_GPU += tests/test-pics-gpu-weights tests/test-pics-gpu-noncart-weights tests/test-pics-gpu-llr
TESTS_GPU += tests/test-pics-multigpu

ifeq ($(MPI), 1)
TESTS_GPU += tests/test-pics-gpu-mpi tests/test-pics-gpu-noncart-weights-mpi tests/test-pics-cart-mpi-gpu
endif

ifeq ($(OMP), 1)
TESTS_GPU += tests/test-pics-gpu-omp tests/test-pics-gpu-noncart-weights-omp
endif


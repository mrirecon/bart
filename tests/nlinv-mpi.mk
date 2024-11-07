tests/test-nlinv-cart-mpi: bart pics copy nrmse $(TESTS_OUT)/ksp_usamp_1.ra $(TESTS_OUT)/sens_1.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)												;\
	$(TOOLDIR)/copy $(TESTS_OUT)/ksp_usamp_1.ra shepplogan_coil_ksp										;\
	OMP_NUM_THREADS=1 mpirun -n 2 $(ROOTDIR)/bart --md-split-mpi-dims=8 nlinv shepplogan_coil_ksp reco coils				;\
	                              $(ROOTDIR)/bart                       nlinv shepplogan_coil_ksp reco_ref coils_ref			;\
	$(TOOLDIR)/nrmse -t 1e-5 reco_ref reco													;\
	$(TOOLDIR)/nrmse -t 1e-5 coils_ref coils												;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-nlinv-noncart-mpi: bart traj phantom ones pics nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP); export OMP_NUM_THREADS=1									;\
	$(TOOLDIR)/traj -r -x128 -o2. -y64 traj.ra 												;\
	$(TOOLDIR)/phantom -s4 -t traj.ra ksp													;\
	mpirun -n 2 $(ROOTDIR)/bart -S --md-split-mpi-dims=8 nlinv -RT:7:0:0.01 -t traj.ra ksp reco1 col1					;\
		    $(ROOTDIR)/bart                          nlinv -RT:7:0:0.01 -t traj.ra ksp reco2 col2					;\
	$(TOOLDIR)/nrmse -t 0.001 reco1 reco2													;\
	$(TOOLDIR)/nrmse -t 0.001 col1 col2													;\
	rm *.cfl *.ra *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_MPI += tests/test-nlinv-cart-mpi tests/test-nlinv-noncart-mpi



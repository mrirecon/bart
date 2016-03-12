
# compare with FFT on a Cartesian grid

tests/test-nufft-forward: traj nufft reshape nrmse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan_fftu.ra
	$(TOOLDIR)/traj -x128 -y128 $(TESTS_TMP)/traj.ra
	$(TOOLDIR)/nufft $(TESTS_TMP)/traj.ra $(TESTS_OUT)/shepplogan.ra $(TESTS_TMP)/shepplogan_ksp2.ra
	$(TOOLDIR)/reshape 7 128 128 1 $(TESTS_TMP)/shepplogan_ksp2.ra $(TESTS_TMP)/shepplogan_ksp3.ra
	$(TOOLDIR)/nrmse -t 0.01 $(TESTS_OUT)/shepplogan_fftu.ra $(TESTS_TMP)/shepplogan_ksp3.ra
	rm $(TESTS_TMP)/*.ra
	touch $@



# test adjoint using definition

tests/test-nufft-adjoint: zeros noise reshape traj nufft fmac nrmse
	$(TOOLDIR)/zeros 3 128 128 1 $(TESTS_TMP)/z.ra
	$(TOOLDIR)/noise -s123 $(TESTS_TMP)/z.ra $(TESTS_TMP)/n1.ra
	$(TOOLDIR)/noise -s321 $(TESTS_TMP)/z.ra $(TESTS_TMP)/n2b.ra
	$(TOOLDIR)/reshape 7 1 128 128 $(TESTS_TMP)/n2b.ra $(TESTS_TMP)/n2.ra
	$(TOOLDIR)/traj -r -x128 -y128 $(TESTS_TMP)/traj.ra
	$(TOOLDIR)/nufft $(TESTS_TMP)/traj.ra $(TESTS_TMP)/n1.ra $(TESTS_TMP)/k.ra
	$(TOOLDIR)/nufft -a $(TESTS_TMP)/traj.ra $(TESTS_TMP)/n2.ra $(TESTS_TMP)/x.ra
	$(TOOLDIR)/fmac -C -s7 $(TESTS_TMP)/n1.ra $(TESTS_TMP)/x.ra $(TESTS_TMP)/s1.ra
	$(TOOLDIR)/fmac -C -s7 $(TESTS_TMP)/k.ra $(TESTS_TMP)/n2.ra $(TESTS_TMP)/s2.ra
	$(TOOLDIR)/nrmse -t 0.00001 $(TESTS_TMP)/s1.ra $(TESTS_TMP)/s2.ra
	rm $(TESTS_TMP)/*.ra
	touch $@



# test inverse using definition

tests/test-nufft-inverse: traj scale phantom nufft nrmse
	$(TOOLDIR)/traj -r -x256 -y201 $(TESTS_TMP)/traj.ra
	$(TOOLDIR)/scale 0.5 $(TESTS_TMP)/traj.ra $(TESTS_TMP)/traj2.ra
	$(TOOLDIR)/phantom $(TESTS_TMP)/x.ra
	$(TOOLDIR)/nufft $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/x.ra $(TESTS_TMP)/ksp.ra
	$(TOOLDIR)/nufft -i $(TESTS_TMP)/traj2.ra $(TESTS_TMP)/ksp.ra $(TESTS_TMP)/reco.ra
	$(TOOLDIR)/nrmse -t 0.12 $(TESTS_TMP)/x.ra $(TESTS_TMP)/reco.ra
	rm $(TESTS_TMP)/*.ra
	touch $@



# test toeplitz by comparing to non-toeplitz

tests/test-nufft-toeplitz: traj phantom nufft nrmse
	$(TOOLDIR)/traj -r -x128 -y128 $(TESTS_TMP)/traj.ra
	$(TOOLDIR)/phantom -k -t $(TESTS_TMP)/traj.ra $(TESTS_TMP)/ksp
	$(TOOLDIR)/nufft -i $(TESTS_TMP)/traj.ra $(TESTS_TMP)/ksp $(TESTS_TMP)/reco1.ra
	$(TOOLDIR)/nufft -i -t $(TESTS_TMP)/traj.ra $(TESTS_TMP)/ksp $(TESTS_TMP)/reco2.ra
	$(TOOLDIR)/nrmse -t 0.01 $(TESTS_TMP)/reco1.ra $(TESTS_TMP)/reco2.ra
	rm $(TESTS_TMP)/*.ra
	touch $@


TESTS += tests/test-nufft-forward tests/test-nufft-adjoint tests/test-nufft-inverse tests/test-nufft-toeplitz





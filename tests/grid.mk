
$(TESTS_OUT)/grid_t: traj
	$(TOOLDIR)/traj -x 128 -y 129 $@

tests/test-grid_traj: grid extract nrmse $(TESTS_OUT)/grid_t
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/grid -t $(TESTS_OUT)/grid_t -T 7 grid_tt ;\
	$(TOOLDIR)/extract 0 0 3 10 0 1 grid_tt grid_tt_e ;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/grid_t grid_tt_e ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-grid_cart-traj: traj grid extract nrmse $(TESTS_OUT)/grid_t
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/grid -k -D 128:129 grid_k ;\
	$(TOOLDIR)/extract 0 0 3 grid_k grid_k_e ;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/grid_t grid_k_e ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-grid_cart-index: grid extract nrmse ones index saxpy
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/grid -D 5:1:1 --b1=2:0:0 grid ;\
	$(TOOLDIR)/index 1 5 i ;\
	$(TOOLDIR)/ones 2 1 5 ones ;\
	$(TOOLDIR)/saxpy -- -2 ones i i_grid ;\
	$(TOOLDIR)/extract 0 0 1 grid grid_0 ;\
	$(TOOLDIR)/nrmse -t 0 i_grid grid_0 ;\
	$(TOOLDIR)/grid -D 1:5:1 --b2=0:2:0 grid ;\
	$(TOOLDIR)/index 2 5 i ;\
	$(TOOLDIR)/ones 3 1 1 5 ones ;\
	$(TOOLDIR)/saxpy -- -2 ones i i_grid ;\
	$(TOOLDIR)/extract 0 1 2 grid grid_0 ;\
	$(TOOLDIR)/nrmse -t 0 i_grid grid_0 ;\
	$(TOOLDIR)/grid -D 1:1:5 --b3=0:0:2 grid ;\
	$(TOOLDIR)/index 3 5 i ;\
	$(TOOLDIR)/ones 4 1 1 1 5 ones ;\
	$(TOOLDIR)/saxpy -- -2 ones i i_grid ;\
	$(TOOLDIR)/extract 0 2 3 grid grid_0 ;\
	$(TOOLDIR)/nrmse -t 0 i_grid grid_0 ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-grid_traj_compat: grid nrmse $(TESTS_OUT)/grid_t
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/grid -t $(TESTS_OUT)/grid_t -T 7 traj ;\
	$(TOOLDIR)/grid -t traj ttraj ;\
	$(TOOLDIR)/nrmse -t 0 traj ttraj ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-grid_traj tests/test-grid_cart-traj tests/test-grid_cart-index tests/test-grid_traj_compat


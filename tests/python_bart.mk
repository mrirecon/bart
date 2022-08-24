

tests/test-python-bart: $(TOOLDIR)/python/bart.py bart
	PYTHONPATH=$(TOOLDIR)/python python3 -c "import bart; bart.bart(0,'version -V')"
	touch $@

tests/test-python-bart-io: $(TOOLDIR)/python/bart.py $(TESTS_OUT)/shepplogan.ra bart nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                                            ;\
	PYTHONPATH=$(TOOLDIR)/python python3 -c "import bart; import cfl;			\
			pht=bart.bart(1, 'phantom'); cfl.writecfl('shepplogan_py', pht);"	;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan.ra shepplogan_py			;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-python-bart-io-kwargs: $(TOOLDIR)/python/bart.py $(TESTS_OUT)/shepplogan_ksp.ra bart traj reshape nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)                                            ;\
	$(TOOLDIR)/traj traj                                                                 	;\
	PYTHONPATH=$(TOOLDIR)/python python3 -c "import bart; import cfl; 			\
		trajpy=cfl.readcfl('traj'); pht=bart.bart(1,'phantom -k',t=trajpy); 		\
		cfl.writecfl('shepplogan_ksp_py', pht);"					;\
	$(TOOLDIR)/reshape 7 128 128 1 shepplogan_ksp_py shepplogan_ksp3			;\
	$(TOOLDIR)/nrmse -t 0. $(TESTS_OUT)/shepplogan_ksp.ra shepplogan_ksp3 			;\
	rm *.cfl *.hdr ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_PYTHON += tests/test-python-bart
TESTS_PYTHON += tests/test-python-bart-io tests/test-python-bart-io-kwargs


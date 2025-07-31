


tests/test-pole: ones phasepole fmac nrmse conj
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phasepole -s --center 0.25:0.25:0. -x64:64:1 pole.ra				;\
	$(TOOLDIR)/conj pole.ra pole.ra								;\
	$(TOOLDIR)/phasepole -s --center 0.15:0.25:0. -x64:64:1 pole2.ra				;\
	$(TOOLDIR)/fmac pole.ra pole2.ra pole.ra						;\
	$(TOOLDIR)/phasepole pole.ra det_pole.ra							;\
	$(TOOLDIR)/nrmse -t 0.05 pole.ra det_pole.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-pole

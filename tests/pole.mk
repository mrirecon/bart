


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

tests/test-pole-nlinv: ones phasepole nlinv fmac nrmse conj zeros join $(TESTS_OUT)/shepplogan_coil_ksp.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
	$(TOOLDIR)/phasepole -s --center -0.25:0.05:0. -x128:128:1 o.ra				;\
	$(TOOLDIR)/zeros 4 128 128 1 8 z.ra							;\
	$(TOOLDIR)/join 3 o.ra z.ra i.ra							;\
	$(TOOLDIR)/nlinv --sens-os=1.5 --phase-pole=8 -i10 -Ii.ra $(TESTS_OUT)/shepplogan_coil_ksp.ra img.ra col.ra	;\
	$(TOOLDIR)/phasepole col.ra cpole.ra 							;\
	$(TOOLDIR)/ones 2 128 128 o.ra								;\
	$(TOOLDIR)/nrmse -t 0. cpole.ra o.ra							;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-pole
TESTS += tests/test-pole-nlinv


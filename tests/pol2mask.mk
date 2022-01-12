


tests/test-pol2mask: index zexp creal scale ones saxpy join pol2mask phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/index 1 6282 i0.ra							;\
	$(TOOLDIR)/scale 0.001 i0.ra i1.ra						;\
	$(TOOLDIR)/zexp -i i1.ra i2.ra							;\
	$(TOOLDIR)/creal i2.ra i2r.ra							;\
	$(TOOLDIR)/scale -- -1.i i2.ra ii2.ra						;\
	$(TOOLDIR)/creal ii2.ra i2i.ra							;\
	$(TOOLDIR)/ones 2 1 6282 o.ra							;\
	$(TOOLDIR)/saxpy 1. o.ra i2r.ra i3r.ra						;\
	$(TOOLDIR)/saxpy 1. o.ra i2i.ra i3i.ra						;\
	$(TOOLDIR)/join 0 i3r.ra i3i.ra i3.ra						;\
	$(TOOLDIR)/scale -- 50. i3.ra i4.ra						;\
	$(TOOLDIR)/pol2mask i4.ra i5.ra							;\
	$(TOOLDIR)/phantom -c -x100 r.ra						;\
	$(TOOLDIR)/nrmse -t 0.05 r.ra i5.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-pol2mask2: index zexp creal scale ones saxpy join pol2mask resize nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/index 1 6 i1.ra							;\
	$(TOOLDIR)/zexp -i i1.ra i2.ra							;\
	$(TOOLDIR)/creal i2.ra i2r.ra							;\
	$(TOOLDIR)/scale -- -1.i i2.ra ii2.ra						;\
	$(TOOLDIR)/creal ii2.ra i2i.ra							;\
	$(TOOLDIR)/ones 2 1 6 o.ra							;\
	$(TOOLDIR)/saxpy 25. o.ra i2r.ra i3r.ra						;\
	$(TOOLDIR)/saxpy 25. o.ra i2i.ra i3i.ra						;\
	$(TOOLDIR)/join 0 i3r.ra i3i.ra i3.ra						;\
	$(TOOLDIR)/pol2mask -X50 -Y50 i3.ra i5.ra					;\
	$(TOOLDIR)/ones 1 1 o.ra							;\
	$(TOOLDIR)/resize -c 0 50 1 50 o.ra r.ra					;\
	$(TOOLDIR)/nrmse -t 0. r.ra i5.ra 						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@




TESTS += tests/test-pol2mask tests/test-pol2mask2




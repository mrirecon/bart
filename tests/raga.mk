
tests/test-raga: raga extract vec transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga 377 i.ra				;\
	$(TOOLDIR)/extract 2 0 5 i.ra ie.ra			;\
	$(TOOLDIR)/vec 0 233 89 322 178 v.ra			;\
	$(TOOLDIR)/transpose 0 2 v.ra v2.ra			;\
	$(TOOLDIR)/nrmse -t 0. ie.ra v2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-raga-tiny: raga extract vec transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga -s 7 419 i.ra				;\
	$(TOOLDIR)/extract 2 0 5 i.ra ie.ra			;\
	$(TOOLDIR)/vec 0 55 110 165 220 v.ra			;\
	$(TOOLDIR)/transpose 0 2 v.ra v2.ra			;\
	$(TOOLDIR)/nrmse -t 0. ie.ra v2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-raga-inc: raga nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga -r 55 419 i.ra				;\
	$(TOOLDIR)/raga -s 7 419 i2.ra				;\
	$(TOOLDIR)/nrmse -t 0. i.ra i2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-raga-single: raga extract vec transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga -s 7 --no-double-base 838 i.ra		;\
	$(TOOLDIR)/extract 2 0 5 i.ra ie.ra			;\
	$(TOOLDIR)/raga -s 7 419 i2.ra				;\
	$(TOOLDIR)/extract 2 0 5 i2.ra i2e.ra			;\
	$(TOOLDIR)/nrmse -t 0.000001 ie.ra i2e.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-raga tests/test-raga-tiny
TESTS += tests/test-raga-inc tests/test-raga-single

# Depreciated Part!
tests/test-raga-old: raga traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/raga 377 i.ra						;\
	$(TOOLDIR)/traj -y 377 -s 1 --double-base --raga-index-file i2.ra t.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 i.ra i2.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-raga-old-tiny-single: raga traj nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/raga --no-double-base -s 7 838 i.ra			;\
	$(TOOLDIR)/traj -y 838 -s 7 --raga-index-file i2.ra t.ra	;\
	$(TOOLDIR)/nrmse -t 0.000001 i.ra i2.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-raga-old tests/test-raga-old-tiny-single


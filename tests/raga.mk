
tests/test-raga: raga extract vec transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga 377 i.ra				;\
	$(TOOLDIR)/extract 2 0 5 i.ra ie.ra			;\
	$(TOOLDIR)/vec 0 233 89 322 178 v.ra			;\
	$(TOOLDIR)/transpose 0 2 v.ra v2.ra			;\
	$(TOOLDIR)/nrmse -t 0. ie.ra v2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-raga


tests/test-raga-tiny: raga extract vec transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga -s 7 419 i.ra				;\
	$(TOOLDIR)/extract 2 0 5 i.ra ie.ra			;\
	$(TOOLDIR)/vec 0 55 110 165 220 v.ra			;\
	$(TOOLDIR)/transpose 0 2 v.ra v2.ra			;\
	$(TOOLDIR)/nrmse -t 0. ie.ra v2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-raga-tiny


tests/test-raga-inc: raga nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga -r 55 419 i.ra				;\
	$(TOOLDIR)/raga -s 7 419 i2.ra				;\
	$(TOOLDIR)/nrmse -t 0. i.ra i2.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-raga-inc


tests/test-raga-single: raga extract vec transpose nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/raga -s 7 --no-double-base 838 i.ra		;\
	$(TOOLDIR)/extract 2 0 5 i.ra ie.ra			;\
	$(TOOLDIR)/raga -s 7 419 i2.ra				;\
	$(TOOLDIR)/extract 2 0 5 i2.ra i2e.ra			;\
	$(TOOLDIR)/nrmse -t 0.000001 ie.ra i2e.ra		;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@-

 TESTS += tests/test-raga-single


tests/test-raga-multislice: raga vec transpose bin circshift join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/raga -s 1 -m 3 13 ind_ms.ra				;\
	$(TOOLDIR)/raga -s 1      13 ind_ss.ra				;\
	$(TOOLDIR)/vec 0 3 6 9 12 2 5 8 11 1 4 7 10 ind1_t.ra		;\
	$(TOOLDIR)/transpose 0 2 ind1_t.ra ind1.ra			;\
	$(TOOLDIR)/bin -o ind1.ra ind_ss.ra ind_s1.ra			;\
	$(TOOLDIR)/circshift 2 4 ind_s1.ra ind_s2.ra			;\
	$(TOOLDIR)/circshift 2 4 ind_s2.ra ind_s3.ra			;\
	$(TOOLDIR)/join 13 ind_s1.ra ind_s2.ra ind_s3.ra ind_ref.ra	;\
	$(TOOLDIR)/nrmse -t 0. ind_ms.ra ind_ref.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

 TESTS += tests/test-raga-multislice



tests/test-raga-multi-inversion: raga circshift join nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/raga -s 1 -i 3 13 ind_mi.ra				;\
	$(TOOLDIR)/raga -s 1      13 ind_i1.ra				;\
	$(TOOLDIR)/circshift 2 12 ind_i1.ra ind_i2.ra			;\
	$(TOOLDIR)/circshift 2 11 ind_i1.ra ind_i3.ra			;\
	$(TOOLDIR)/join 15 ind_i1.ra ind_i2.ra ind_i3.ra ind_ref.ra	;\
	$(TOOLDIR)/nrmse -t 0. ind_mi.ra ind_ref.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

 TESTS += tests/test-raga-multi-inversion


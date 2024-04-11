

tests/test-opts-mix-args: phantom nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/phantom -x 32 -s 4 -B -r 875293 phan1.ra 	;\
	$(TOOLDIR)/phantom -x 32 phan2.ra -s 4 -B -r 875293 	;\
	$(TOOLDIR)/nrmse -t 0.0 phan1.ra phan2.ra		;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@



tests/test-opts-bart-loop: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(ROOTDIR)/bart phantom phan.ra 				;\
	$(ROOTDIR)/bart                fft -u -i 2 phan.ra ksp1.ra	;\
	$(ROOTDIR)/bart -l 1 -rphan.ra fft 2 -i phan.ra ksp2.ra -u	;\
	$(ROOTDIR)/bart nrmse -t 1e-6 ksp1.ra ksp2.ra			;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-opts-bart-loop tests/test-opts-mix-args

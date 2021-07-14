
tests/test-morphop-dilation-erosion: phantom morphop nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -g 3 ori.ra					;\
	$(TOOLDIR)/morphop -e 9 ori.ra redu.ra					;\
	$(TOOLDIR)/morphop -d 9 redu.ra rec.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 ori.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-morphop-dilation-erosion-large: phantom morphop nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x128 -g 3 ori.ra					;\
	$(TOOLDIR)/morphop -e 51 ori.ra redu.ra				;\
	$(TOOLDIR)/morphop -d 51 redu.ra rec.ra				;\
	$(TOOLDIR)/nrmse -t 0.000001 ori.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-morphop-opening: phantom morphop nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -g 2 ori.ra					;\
	$(TOOLDIR)/morphop -e 9 ori.ra tmp.ra					;\
	$(TOOLDIR)/morphop -d 9 tmp.ra rec.ra					;\
	$(TOOLDIR)/morphop -o 9 ori.ra rec2.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 rec2.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-morphop-closing: phantom morphop nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom -x64 -g 2 ori.ra					;\
	$(TOOLDIR)/morphop -d 9 ori.ra tmp.ra					;\
	$(TOOLDIR)/morphop -e 9 tmp.ra rec.ra					;\
	$(TOOLDIR)/morphop -c 9 ori.ra rec2.ra					;\
	$(TOOLDIR)/nrmse -t 0.000001 rec2.ra rec.ra 				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-morphop-dilation-erosion tests/test-morphop-dilation-erosion-large tests/test-morphop-opening tests/test-morphop-closing


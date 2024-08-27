
tests/test-pipe: phantom copy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP);\
	$(TOOLDIR)/phantom -s3 phantom.ra 			;\
	$(TOOLDIR)/copy -- phantom.ra - |	 		 \
	$(TOOLDIR)/nrmse -t 0 -- - phantom.ra 	;\
	rm *.ra	; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stream-long-header: ones copy show
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	FILENAME=We_need_a_really_long_filename-\
	JoinusnowandsharethesoftwareYoullbefreehackersyoullbefree	;\
	$(TOOLDIR)/ones 1 42 $$FILENAME					;\
	$(TOOLDIR)/copy -- $$FILENAME -					|\
	$(TOOLDIR)/show -m -- -						;\
	rm $$FILENAME.{cfl,hdr}; cd ..; rmdir $(TESTS_TMP)
	touch $@

.PHONY: tests/test-stream
tests/test-stream: tests/test-pipe tests/test-stream-long-header

TESTS += tests/test-stream


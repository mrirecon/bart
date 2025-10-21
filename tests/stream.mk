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
	rm $$FILENAME.{cfl,hdr}; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream1: phantom copy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/phantom -s3 phantom.ra 		;\
	$(TOOLDIR)/copy --stream 9 -- phantom.ra - |	 \
	$(TOOLDIR)/nrmse -t 0 -- - phantom.ra 		;\
	rm *.ra ; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream2: phantom copy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/phantom -s3 -- phantom.ra						;\
	$(TOOLDIR)/phantom -s3 -- - | $(TOOLDIR)/copy --stream 9 -- - phantom2.ra	;\
	$(TOOLDIR)/nrmse -t 0 phantom.ra phantom2.ra					;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream3: phantom copy nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/phantom -s3 -- phantom.ra		;\
	$(TOOLDIR)/copy --stream 1 -- phantom.ra - | 	 \
	$(TOOLDIR)/copy --stream 8 -- - phantom2.ra	;\
	$(TOOLDIR)/nrmse -t 0 phantom.ra phantom2.ra	;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream4: phantom copy nrmse repmat
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)								;\
	$(TOOLDIR)/phantom - | $(TOOLDIR)/repmat 10 100 - phantom.ra						;\
	$(TOOLDIR)/copy --stream 1024 phantom.ra - | $(TOOLDIR)/copy - - | $(TOOLDIR)/nrmse -t0 - phantom.ra	;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream5: phantom copy nrmse repmat bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)									;\
	$(TOOLDIR)/phantom - | $(TOOLDIR)/repmat 10 100 - phantom.ra							;\
	$(TOOLDIR)/copy --stream 1024 phantom.ra - | $(ROOTDIR)/bart -r - copy - - | $(TOOLDIR)/nrmse -t0 - phantom.ra	;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream-loop: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)									;\
	$(ROOTDIR)/bart mandelbrot -s 64 -n8 -I -- - | $(ROOTDIR)/bart -l4 -e8 resize -c -- 0 32 - - | 		\
	$(ROOTDIR)/bart copy -- - ret1.ra										;\
	$(ROOTDIR)/bart mandelbrot -s 64 -n8 -I tmp.ra								;\
	$(ROOTDIR)/bart resize -c 0 32 tmp.ra ret2.ra									;\
	$(ROOTDIR)/bart nrmse -t 0 ret1.ra ret2.ra									;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream-loop-ref: bart
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)									;\
	$(ROOTDIR)/bart mandelbrot -s 64 -n8 -I -- - | $(ROOTDIR)/bart -t3 -r - resize -c -- 0 32 - - | 		\
	$(ROOTDIR)/bart copy -- - ret1.ra										;\
	$(ROOTDIR)/bart mandelbrot -s 64 -n8 -I tmp.ra									;\
	$(ROOTDIR)/bart resize -c 0 32 tmp.ra ret2.ra									;\
	$(ROOTDIR)/bart nrmse -t 0 ret1.ra ret2.ra									;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream-binary: phantom copy nrmse trx
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)									;\
	$(TOOLDIR)/phantom -s3 phantom.ra 										;\
	$(TOOLDIR)/copy --stream 9 -- phantom.ra - | \
	$(TOOLDIR)/trx | $(TOOLDIR)/trx | \
	$(TOOLDIR)/nrmse -t 0 -- - phantom.ra 										;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream-binary2: phantom copy nrmse trx
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/phantom -s3 phantom.ra 		;\
	$(TOOLDIR)/trx -i phantom.ra > phantom.bstrm;\
	$(TOOLDIR)/trx < phantom.bstrm | \
	$(TOOLDIR)/nrmse -t 0 -- - phantom.ra ;\
	rm *.ra	; rm phantom.bstrm; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stream-binary3: phantom copy nrmse trx
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/phantom -s3 phantom.ra 			;\
	$(TOOLDIR)/trx -i phantom.ra > phantom.bstrm		;\
	$(TOOLDIR)/nrmse -t 0 -- - phantom.ra < phantom.bstrm 	;\
	rm *.ra	; rm phantom.bstrm; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stream-binary4: phantom copy nrmse trx
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/phantom -s3 phantom.ra 		;\
	$(TOOLDIR)/copy --stream 9 -- phantom.ra - | \
	$(TOOLDIR)/trx | \
	$(TOOLDIR)/nrmse -t 0 -- phantom.ra - 		;\
	rm *.ra	; cd .. ; rmdir --ignore-fail-on-non-empty $(TESTS_TMP)
	touch $@

tests/test-stream-binary5: bart copy nrmse trx
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(ROOTDIR)/bart --stream-bin-out phantom - > phantom.bstrm	;\
	$(TOOLDIR)/copy - phantom.ra < phantom.bstrm			;\
	$(TOOLDIR)/nrmse -t 0 phantom.ra - < phantom.bstrm		;\
	rm *.ra	; rm phantom.bstrm; cd .. ; rmdir $(TESTS_TMP)
	touch $@


.PHONY: tests/test-stream
tests/test-stream: tests/test-pipe tests/test-stream1 tests/test-stream2 tests/test-stream3 tests/test-stream4 tests/test-stream5 \
	tests/test-stream-loop tests/test-stream-loop-ref tests/test-stream-binary tests/test-stream-binary2 \
	tests/test-stream-binary3 tests/test-stream-binary4 tests/test-stream-binary5


TESTS += tests/test-stream
TESTS += tests/test-stream-long-header


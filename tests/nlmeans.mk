
tests/test-nlmeans: phantom noise nlmeans nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/phantom phantom.ra 						;\
	$(TOOLDIR)/noise -r -n 0.005 phantom.ra noisy_phantom.ra 		;\
	$(TOOLDIR)/nlmeans -H.22 3 noisy_phantom.ra smooth_phantom.ra 		;\
	$(TOOLDIR)/nrmse -t 0.09 phantom.ra smooth_phantom.ra 			;\
	rm *.ra	; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-nlmeans


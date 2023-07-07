tests/test-nlmeans: phantom noise nlmeans nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)							;\
	$(TOOLDIR)/phantom phantom 												;\
	$(TOOLDIR)/noise -r -n 0.005 phantom noisy_phantom 						;\
	$(TOOLDIR)/nlmeans -H.22 3 noisy_phantom smooth_phantom 				;\
	$(TOOLDIR)/nrmse -t 0.09 phantom smooth_phantom 						;\
	rm phantom.{hdr,cfl} noisy_phantom.{hdr,cfl} smooth_phantom.{hdr,cfl}	;\
	rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-nlmeans

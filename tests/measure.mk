tests/test-measure-ideal-mse: measure $(TESTS_OUT)/shepplogan.ra
	set -e															;\
	if [ "0.000000e+00" != $$($(TOOLDIR)/measure --mse $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan.ra) ]		;\
	then															 \
		false														;\
	fi
	touch $@

tests/test-measure-ideal-mse-mag: measure $(TESTS_OUT)/shepplogan_coil.ra
	set -e															;\
	if [ "0.000000e+00" != $$($(TOOLDIR)/measure --mse-mag $(TESTS_OUT)/shepplogan_coil.ra $(TESTS_OUT)/shepplogan_coil.ra) ]		;\
	then															 \
		false														;\
	fi
	touch $@

tests/test-measure-ideal-ssim: measure $(TESTS_OUT)/shepplogan.ra
	set -e															;\
	if [ "1.000000e+00" != $$($(TOOLDIR)/measure --ssim $(TESTS_OUT)/shepplogan.ra $(TESTS_OUT)/shepplogan.ra) ]		;\
	then															 \
		false														;\
	fi
	touch $@
	


TESTS += tests/test-measure-ideal-mse
TESTS += tests/test-measure-ideal-mse-mag
TESTS += tests/test-measure-ideal-ssim

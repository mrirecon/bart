

# check if bart can parse its own version
tests/test-version: version
	$(TOOLDIR)/version -t `cat $(ROOTDIR)/version.txt`
	touch $@

TESTS += tests/test-version

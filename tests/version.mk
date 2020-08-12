

# check if bart can parse its own version
tests/test-version: version
	$(TOOLDIR)/version -t `cat $(TOOLDIR)/version.txt`
	touch $@

TESTS += tests/test-version

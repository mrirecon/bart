


tests/test-whiten: zeros ones noise join whiten std nrmse show
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/zeros 1 4096 z.ra							;\
	$(TOOLDIR)/noise -s 1 -n 1 z.ra n1.ra						;\
	$(TOOLDIR)/noise -s 2 -n 2 z.ra n2.ra						;\
	$(TOOLDIR)/noise -s 3 -n 3 z.ra n3.ra						;\
	$(TOOLDIR)/join 3 n1.ra n2.ra n3.ra n.ra					;\
	$(TOOLDIR)/ones 1 4096 o.ra							;\
	$(TOOLDIR)/noise -s 1 -n 1 o.ra s1.ra						;\
	$(TOOLDIR)/noise -s 2 -n 2 o.ra s2.ra						;\
	$(TOOLDIR)/noise -s 3 -n 3 o.ra s3.ra						;\
	$(TOOLDIR)/join 3 s1.ra s2.ra s3.ra s.ra					;\
	$(TOOLDIR)/whiten s.ra n.ra w.ra						;\
	$(TOOLDIR)/std 7 w.ra d.ra							;\
	$(TOOLDIR)/ones 4 1 1 1 3 o.ra							;\
	$(TOOLDIR)/nrmse -t 0.0002 d.ra o.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-whiten


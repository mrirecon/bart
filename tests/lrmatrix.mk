

# low-rank + sparse
tests/test-lrmatrix: lrmatrix nrmse ones slice
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 2 10 10 o.ra							;\
	$(TOOLDIR)/lrmatrix -d -s o.ra x.ra						;\
	$(TOOLDIR)/slice 12 1 x.ra y.ra							;\
	$(TOOLDIR)/nrmse -t 0.0005 o.ra y.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


# low-rank + sparse
tests/test-lrmatrix-sparse-denoise: lrmatrix noise nrmse ones slice
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 2 10 10 o.ra							;\
	$(TOOLDIR)/noise -n0.01 o.ra n.ra						;\
	$(TOOLDIR)/lrmatrix -d -s n.ra x.ra						;\
	$(TOOLDIR)/slice 12 1 x.ra y.ra							;\
	$(TOOLDIR)/nrmse -t 0.05 o.ra y.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


# low-rank + sparse
tests/test-lrmatrix-sparse-recover: lrmatrix noise nrmse ones resize transpose fmac threshold slice
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 10 o.ra							;\
	$(TOOLDIR)/resize -c 0 20 o.ra o2.ra						;\
	$(TOOLDIR)/transpose 0 1 o2.ra o2T.ra						;\
	$(TOOLDIR)/fmac o2.ra o2T.ra m2.ra						;\
	$(TOOLDIR)/ones 2 20 20 p.ra							;\
	$(TOOLDIR)/noise -S0.5 p.ra p2.ra						;\
	$(TOOLDIR)/threshold -B 0.5 p2.ra p3.ra						;\
	$(TOOLDIR)/fmac m2.ra p3.ra u2.ra						;\
	$(TOOLDIR)/lrmatrix -s u2.ra x.ra						;\
	$(TOOLDIR)/slice 12 1 x.ra y.ra							;\
	$(TOOLDIR)/nrmse -t 0.001 m2.ra y.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


# low-rank matrix recovery
tests/test-lrmatrix-recover: lrmatrix noise nrmse ones resize transpose fmac threshold
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 10 o.ra							;\
	$(TOOLDIR)/resize -c 0 20 o.ra o2.ra						;\
	$(TOOLDIR)/transpose 0 1 o2.ra o2T.ra						;\
	$(TOOLDIR)/fmac o2.ra o2T.ra m2.ra						;\
	$(TOOLDIR)/ones 2 20 20 p.ra							;\
	$(TOOLDIR)/noise -S0.5 p.ra p2.ra						;\
	$(TOOLDIR)/threshold -B 0.5 p2.ra p3.ra						;\
	$(TOOLDIR)/fmac m2.ra p3.ra u2.ra						;\
	$(TOOLDIR)/lrmatrix -k20 -o y.ra u2.ra x.ra					;\
	$(TOOLDIR)/nrmse -t 0.00005 m2.ra y.ra						;\
	$(TOOLDIR)/nrmse -t 0.00005 m2.ra x.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# locally low-rank denoising, FIXME: this does not denoise
tests/test-lrmatrix-denoise: lrmatrix noise nrmse ones resize transpose fmac slice
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 10 o.ra							;\
	$(TOOLDIR)/resize -c 0 20 o.ra o2.ra						;\
	$(TOOLDIR)/transpose 0 1 o2.ra o2T.ra						;\
	$(TOOLDIR)/fmac o2.ra o2T.ra m2.ra						;\
	$(TOOLDIR)/noise -n 0.001 m2.ra n2.ra						;\
	$(TOOLDIR)/lrmatrix -p100. -d -l10 -N -o y.ra n2.ra x.ra			;\
	$(TOOLDIR)/fmac -s 4096 x.ra x3.ra						;\
	$(TOOLDIR)/nrmse -t 0.008 n2.ra x3.ra						;\
	$(TOOLDIR)/slice 12 0 x.ra x2.ra						;\
	$(TOOLDIR)/nrmse -t 0. y.ra x2.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@



# locally low-rank matrix recovery
tests/test-lrmatrix-recover2: lrmatrix noise nrmse ones resize transpose fmac threshold
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
	$(TOOLDIR)/ones 1 10 o.ra							;\
	$(TOOLDIR)/resize -c 0 20 o.ra o2.ra						;\
	$(TOOLDIR)/transpose 0 1 o2.ra o2T.ra						;\
	$(TOOLDIR)/fmac o2.ra o2T.ra m2.ra						;\
	$(TOOLDIR)/ones 2 20 20 p.ra							;\
	$(TOOLDIR)/noise -S0.5 p.ra p2.ra						;\
	$(TOOLDIR)/threshold -B 0.5 p2.ra p3.ra						;\
	$(TOOLDIR)/fmac m2.ra p3.ra u2.ra						;\
	$(TOOLDIR)/lrmatrix -p100. -l5 -o y.ra u2.ra x.ra				;\
	$(TOOLDIR)/nrmse -t 0.007 m2.ra y.ra						;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-lrmatrix tests/test-lrmatrix-denoise
TESTS += tests/test-lrmatrix-sparse-recover tests/test-lrmatrix-sparse-denoise
TESTS += tests/test-lrmatrix-recover tests/test-lrmatrix-recover2



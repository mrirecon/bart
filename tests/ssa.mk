


tests/test-ssa-pca: traj phantom resize squeeze svd transpose ssa cabs nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra					;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra					;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra					;\
		$(TOOLDIR)/squeeze k1.ra kx.ra						;\
		$(TOOLDIR)/svd kx.ra u.ra s.ra vh.ra					;\
		$(TOOLDIR)/ssa -w1 -m0 -n0 kx.ra eof.ra					;\
		$(TOOLDIR)/cabs u.ra uabs.ra						;\
		$(TOOLDIR)/cabs eof.ra eofabs.ra					;\
		$(TOOLDIR)/resize 1 4 uabs.ra utest.ra					;\
		$(TOOLDIR)/resize 1 4 eofabs.ra eoftest.ra				;\
		$(TOOLDIR)/nrmse -t 0.00001 utest.ra eoftest.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ssa: traj phantom resize squeeze svd transpose ssa cabs nrmse casorati
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra					;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra					;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra					;\
		$(TOOLDIR)/squeeze k1.ra kx.ra						;\
		$(TOOLDIR)/resize -c 0 59 kx.ra kx1.ra					;\
		$(TOOLDIR)/casorati 0 10 1 8 kx1.ra kcas.ra				;\
		$(TOOLDIR)/svd kcas.ra u.ra s.ra vh.ra					;\
		$(TOOLDIR)/ssa -w10 -m0 -n0 kx.ra eof.ra				;\
		$(TOOLDIR)/cabs u.ra uabs.ra						;\
		$(TOOLDIR)/cabs eof.ra eofabs.ra					;\
		$(TOOLDIR)/resize 1 10 uabs.ra utest.ra					;\
		$(TOOLDIR)/resize 1 10 eofabs.ra eoftest.ra				;\
		$(TOOLDIR)/nrmse -t 0.000015 utest.ra eoftest.ra			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ssa-backprojection: phantom reshape repmat squeeze transpose noise ssa extract cabs nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)						;\
		$(TOOLDIR)/phantom -m -x12 -s8 ph.ra						;\
		$(TOOLDIR)/reshape 1067 1 1 1152 1 32 ph.ra phc.ra				;\
		$(TOOLDIR)/repmat 11 5 phc.ra phc1.ra						;\
		$(TOOLDIR)/reshape 3072 160 1 phc1.ra phc2.ra					;\
		$(TOOLDIR)/squeeze phc2.ra phc10.ra					        ;\
		$(TOOLDIR)/transpose 0 1 phc10.ra phc3.ra					;\
		$(TOOLDIR)/noise -n1e8 phc3.ra ph_noise.ra					;\
		$(TOOLDIR)/ssa -w33 -m0 -z -n0 phc3.ra eof.ra					;\
		$(TOOLDIR)/ssa -w33 -m0 -z -n0 -r10 ph_noise.ra eof2.ra s.ra back.ra		;\
		$(TOOLDIR)/nrmse -t 0.4 phc3.ra back.ra						;\
		$(TOOLDIR)/ssa -w33 -m0 -z -n0 -r5 phc3.ra eof3.ra tmp.ra tmp1.ra		;\
		$(TOOLDIR)/ssa -w33 -m0 -z -n0 -r5 tmp1.ra eof4.ra				;\
		$(TOOLDIR)/extract 1 0 4 eof3.ra eofex.ra					;\
		$(TOOLDIR)/cabs eofex.ra eofex1.ra						;\
		$(TOOLDIR)/extract 1 0 4 eof4.ra eof4ex.ra					;\
		$(TOOLDIR)/cabs eof4ex.ra eof4ex1.ra						;\
		$(TOOLDIR)/nrmse -t 0.05 eofex1.ra eof4ex1.ra					;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


tests/test-ssa-grouping: traj phantom resize squeeze ssa nrmse
	set -e ; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)					;\
		$(TOOLDIR)/traj -x128 -y50 -G -D t.ra					;\
		$(TOOLDIR)/phantom -t t.ra -s8 k.ra					;\
		$(TOOLDIR)/resize -c 1 1 k.ra k1.ra					;\
		$(TOOLDIR)/squeeze k1.ra kx.ra						;\
		$(TOOLDIR)/ssa -w10 -m0 -n0 -z -r -5 kx.ra eof.ra s.ra backr.ra		;\
		$(TOOLDIR)/ssa -w10 -m0 -n0 -z -g -31 kx.ra eofg.ra sg.ra backg.ra	;\
		$(TOOLDIR)/nrmse -t 0.00001 backr.ra backg.ra				;\
		$(TOOLDIR)/ssa -w10 -m0 -n0 -z -r 5 kx.ra eof1.ra s1.ra backr1.ra	;\
		$(TOOLDIR)/ssa -w10 -m0 -n0 -z -g 31 kx.ra eofg1.ra sg1.ra backg1.ra	;\
		$(TOOLDIR)/nrmse -t 0.00001 backr1.ra backg1.ra				;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-ssa-pca tests/test-ssa tests/test-ssa-backprojection

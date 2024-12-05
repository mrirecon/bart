
# Simples pattern: No repeating pattern in time and
# no split information over two dimensions
# tdims: [  3  64  57   1   1   1   1   1   1   1   1   1   1   1   1   1 ]
# ddims: [  1  64  57   8   1   1   1   1   1   1   1   1   1   1   1   1 ]
tests/test-grog: traj phantom grog pics calc nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) 	;\
	$(TOOLDIR)/traj -x 32 -y 57 -o2 -r t.ra 	;\
	$(TOOLDIR)/phantom -k -s 8 -t t.ra k.ra		;\
	$(TOOLDIR)/phantom -x 32 -S 8 s.ra 		;\
	$(TOOLDIR)/calc zround t.ra t2.ra		;\
	$(TOOLDIR)/grog t.ra k.ra t2.ra k2.ra 		;\
	$(TOOLDIR)/pics -e -S -t t.ra k.ra s.ra ref.ra 	;\
	$(TOOLDIR)/pics -e -S -t t2.ra k2.ra s.ra r.ra 	;\
	$(TOOLDIR)/nrmse -t 0.12 ref.ra r.ra 		;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@

# Trajectory repeats in time dim
# tdims: [  3  64  57   1   1   1   1   1   1   1   1   1   1   1   1   1 ]
# ddims: [  1  64  57   8   1   1   1   1   1   1   2   1   1   1   1   1 ]
tests/test-grog-repeat: traj repmat phantom calc grog slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) 	;\
	$(TOOLDIR)/traj -x 32 -y 57 -o2 -r t.ra 	;\
	$(TOOLDIR)/repmat 10 2 t.ra t2.ra 		;\
	$(TOOLDIR)/phantom -k -s 8 -t t2.ra k.ra	;\
	$(TOOLDIR)/phantom -x 32 -S 8 s.ra 		;\
	$(TOOLDIR)/calc zround t.ra tg.ra		;\
	$(TOOLDIR)/grog t.ra k.ra tg.ra kg.ra 		;\
	$(TOOLDIR)/slice 10 0 kg.ra s1.ra		;\
	$(TOOLDIR)/slice 10 1 kg.ra s2.ra		;\
	$(TOOLDIR)/nrmse -t 0. s1.ra s2.ra		;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@


# Trajectory provides unique information in time dim
# tdims: [  3  64  57   1   1   1   1   1   1   1   2   1   1   1   1   1 ]
# ddims: [  1  64  57   8   1   1   1   1   1   1   2   1   1   1   1   1 ]
tests/test-grog-repeat2: traj repmat phantom calc grog slice nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) 	;\
	$(TOOLDIR)/traj -x 32 -y 57 -o2 -r t.ra 	;\
	$(TOOLDIR)/repmat 10 2 t.ra t2.ra 		;\
	$(TOOLDIR)/phantom -k -s 8 -t t2.ra k.ra	;\
	$(TOOLDIR)/phantom -x 32 -S 8 s.ra 		;\
	$(TOOLDIR)/calc zround t2.ra tg.ra		;\
	$(TOOLDIR)/grog t2.ra k.ra tg.ra kg.ra 		;\
	$(TOOLDIR)/slice 10 0 kg.ra s1.ra		;\
	$(TOOLDIR)/slice 10 1 kg.ra s2.ra		;\
	$(TOOLDIR)/nrmse -t 0. s1.ra s2.ra		;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@


# Case with repeated pattern in TIME_DIM, but split pattern INFO_DIM := 9 < TIME_DIM
# tdims: [  3  64  19   1   1   1   1   1   1   3   1   1   1   1   1   1 ]
# ddims: [  1  64  19   8   1   1   1   1   1   3   2   1   1   1   1   1 ]
tests/test-grog-input-dims: traj repmat reshape phantom transpose calc grog nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP) 	;\
	$(TOOLDIR)/traj -x 32 -y 57 -o2 -r t1.ra 	;\
	$(TOOLDIR)/repmat 10 2 t1.ra t2.ra 		;\
	$(TOOLDIR)/reshape 516 19 3 t1.ra t3.ra 	;\
	$(TOOLDIR)/phantom -k -s 8 -t t2.ra k.ra	;\
	$(TOOLDIR)/transpose 2 3 k.ra k1.ra		;\
	$(TOOLDIR)/reshape 1544 19 3 2 k1.ra k2.ra 	;\
	$(TOOLDIR)/transpose 2 3 k2.ra k3.ra		;\
	$(TOOLDIR)/calc zround t1.ra t2g.ra		;\
	$(TOOLDIR)/calc zround t3.ra t3g.ra		;\
	$(TOOLDIR)/grog --calib-spokes=53 t3.ra k3.ra t3g.ra k3g.ra 	;\
	$(TOOLDIR)/grog --calib-spokes=53 t1.ra k.ra t2g.ra kg.ra 	;\
	$(TOOLDIR)/transpose 2 3 kg.ra kg1.ra		;\
	$(TOOLDIR)/reshape 1544 19 3 2 kg1.ra kg2.ra 	;\
	$(TOOLDIR)/transpose 2 3 kg2.ra kg3.ra		;\
	$(TOOLDIR)/nrmse -t 0. kg3.ra k3g.ra 	;\
	rm *.ra; cd .. ; rmdir $(TESTS_TMP)
	touch $@


TESTS += tests/test-grog
TESTS += tests/test-grog-repeat tests/test-grog-repeat2
TESTS += tests/test-grog-input-dims


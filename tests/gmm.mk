

tests/test-gmm: bart ones scale hist grid slice saxpy gmm squeeze nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)		;\
	$(TOOLDIR)/ones 1 1 o 					;\
	$(TOOLDIR)/scale 0.1+0.2i o z 				;\
	$(TOOLDIR)/scale 50. o v				;\
	$(ROOTDIR)/bart -l 4 --end 30000 gmm --sample o z v s	;\
	$(TOOLDIR)/hist -c 1 s h				;\
	$(TOOLDIR)/grid -D 100:100 g				;\
	$(TOOLDIR)/slice 0 0 g g2a				;\
	$(TOOLDIR)/slice 0 1 g g2b				;\
	$(TOOLDIR)/saxpy 1.i g2b g2a g2c			;\
	$(ROOTDIR)/bart -l 7 -r g2c gmm o z v g2c e		;\
	$(TOOLDIR)/squeeze e es					;\
	$(TOOLDIR)/scale 3. es es2				;\
	$(TOOLDIR)/nrmse -t 0.25 es2 h				;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS_SLOW += tests/test-gmm


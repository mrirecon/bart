tests/test-coils-channelselection: coils nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/coils --H2D8C -n 8 -k s8 ;\
	$(TOOLDIR)/coils --H2D8C -b 255 -k sb8 ;\
	$(TOOLDIR)/nrmse -t 0 s8 sb8 ;\
	$(TOOLDIR)/coils --H3D64C -n 15 -k s64 ;\
	$(TOOLDIR)/coils --H3D64C -b 32767 -k sb64 ;\
	$(TOOLDIR)/nrmse -t 0 s64 sb64 ;\
	$(TOOLDIR)/coils --H3D64C -n 64 -k ss64 ;\
	$(TOOLDIR)/coils --H3D64C -k sss64 ;\
	$(TOOLDIR)/nrmse -t 0 ss64 sss64 ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-coils-sens8: coils fft extract nrmse resize $(TESTS_OUT)/coils.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/coils -L s ;\
	$(TOOLDIR)/nrmse -t 0.00000013 $(TESTS_OUT)/coils.ra s ;\
	$(TOOLDIR)/coils -L -k sk ;\
	$(TOOLDIR)/resize -c 0 256 1 256 sk c_skr ;\
	$(TOOLDIR)/fft -i 3 c_skr sr ;\
	$(TOOLDIR)/extract 0 64 192 1 64 192 sr sre ;\
	$(TOOLDIR)/nrmse -t 0.00000013 sre s ;\
	$(TOOLDIR)/nrmse -t 0.00000014 sre $(TESTS_OUT)/coils.ra ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-coils-sens64: coils fft extract nrmse resize phantom
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/phantom -S 8 --coil HEAD_3D_64CH coils_large ;\
	$(TOOLDIR)/coils --H3D64C -L -n 8 s ;\
	$(TOOLDIR)/nrmse -t 0.00000026 coils_large s ;\
	$(TOOLDIR)/coils --H3D64C -L -k -n 8 sk ;\
	$(TOOLDIR)/resize -c 0 256 1 256 2 256 sk c64_skr ;\
	$(TOOLDIR)/fft -i 7 c64_skr sr64 ;\
	$(TOOLDIR)/extract 0 64 192 1 64 192 2 64 192 sr64 sre64 ;\
	$(TOOLDIR)/extract 2 64 65 sre64 sree64 ;\
	$(TOOLDIR)/nrmse -t 0.00000016 sree64 s ;\
	$(TOOLDIR)/nrmse -t 0.00000026 sree64 coils_large ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-coils-grid: coils phantom grid nrmse
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/phantom -S 8 -x 1234 s ;\
	$(TOOLDIR)/grid --b1 1:0:0 --b2 0:1:0 -D 1234:1234 g ;\
	$(TOOLDIR)/coils -t g -L ss ;\
	$(TOOLDIR)/nrmse -t 0.00000013 s ss ;\
	rm *.cfl *.hdr; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-coils-channelselection tests/test-coils-sens8 tests/test-coils-grid

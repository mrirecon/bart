
$(TESTS_OUT)/tetrahedron_cfl: stl
	$(TOOLDIR)/stl --model=TET $@

tests/test-stl-rw-tetrahedron: nrmse stl $(TESTS_OUT)/tetrahedron_cfl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)	;\
	$(TOOLDIR)/stl --input $(TESTS_OUT)/tetrahedron_cfl tet.stl	;\
	$(TOOLDIR)/stl --input tet.stl tet_cfl          ;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/tetrahedron_cfl tet_cfl ;\
	rm *.cfl *.hdr *.stl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-stl-rw-tetrahedron 


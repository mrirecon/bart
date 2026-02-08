
$(TESTS_OUT)/tetrahedron.ra: stl
	$(TOOLDIR)/stl --model=TET $@

$(TESTS_OUT)/ascii_enc.stl:
	echo "solid name" > $@		;\
	echo "facet normal 0 0 1" >> $@	;\
	echo "outer loop" >> $@		;\
	echo "vertex 1 0 0" >> $@	;\
	echo "vertex 0 0 0" >> $@	;\
	echo "vertex 0 1 0" >> $@	;\
	echo "endloop" >> $@		;\
	echo "endfacet" >> $@		;\
	echo "endsolid name" >> $@

tests/test-stl-rw: nrmse stl $(TESTS_OUT)/tetrahedron.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/stl --input $(TESTS_OUT)/tetrahedron.ra tet.stl	;\
	$(TOOLDIR)/stl --input tet.stl tet.ra        			;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/tetrahedron.ra tet.ra 	;\
	rm *.ra *.stl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stl-rw-binary: nrmse stl $(TESTS_OUT)/tetrahedron.ra
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)				;\
	$(TOOLDIR)/stl --binary --input $(TESTS_OUT)/tetrahedron.ra tet.stl	;\
	$(TOOLDIR)/stl --input tet.stl tet.ra        				;\
	$(TOOLDIR)/nrmse -t 0 $(TESTS_OUT)/tetrahedron.ra tet.ra 		;\
	rm *.ra *.stl ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

tests/test-stl-read-ascii: vec join nrmse stl $(TESTS_OUT)/ascii_enc.stl
	set -e; mkdir $(TESTS_TMP) ; cd $(TESTS_TMP)			;\
	$(TOOLDIR)/stl --input $(TESTS_OUT)/ascii_enc.stl tri.ra 	;\
	$(TOOLDIR)/vec -- 1. 0. 0. v0.ra				;\
	$(TOOLDIR)/vec -- 0. 0. 0. v1.ra				;\
	$(TOOLDIR)/vec -- 0. 1. 0. v2.ra				;\
	$(TOOLDIR)/vec -- 0. 0. -1. n.ra				;\
	$(TOOLDIR)/join 1 v0.ra v1.ra v2.ra n.ra tri_ref.ra		;\
	$(TOOLDIR)/nrmse -t 0. tri_ref.ra tri.ra 			;\
	rm *.ra ; cd .. ; rmdir $(TESTS_TMP)
	touch $@

TESTS += tests/test-stl-rw tests/test-stl-read-ascii tests/test-stl-rw-binary


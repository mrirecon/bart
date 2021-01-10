



#bart vec 0 0 1 0 v1
#bart vec 0 0 0 1 v2
#bart vec 0 1 1 1 v3
#bart join 1 v1 v2 v3 v

#bart vec 0 0 1 1 v1
#bart vec 0 1 1 0 v2
#bart vec 0 0 1 0 v3
#bart join 1 v1 v2 v3 v

#bart vec 0 1 1 1 0 1 v1
#bart vec 0 1 0 0 0 0 v2
#bart vec 0 0 0 0 1 1 v3
#bart vec 0 0 1 1 0 1 v4
#bart vec 0 1 0 1 0 1 v5
#bart join 1 v1 v2 v3 v4 v5 v

#bart resize -c 0 300 1 300 v o
#bart conway -n3000 o x


bart vec 0 0 0 1 1 1 0 0 0 1 1 1 0 0 v0
bart vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v1
bart vec 0 1 0 0 0 0 1 0 1 0 0 0 0 1 v2
bart vec 0 1 0 0 0 0 1 0 1 0 0 0 0 1 v3
bart vec 0 1 0 0 0 0 1 0 1 0 0 0 0 1 v4
bart vec 0 0 0 1 1 1 0 0 0 1 1 1 0 0 v5
bart vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0 v6
bart vec 0 0 0 1 1 1 0 0 0 1 1 1 0 0 v7
bart vec 0 1 0 0 0 0 1 0 1 0 0 0 0 1 v8
bart vec 0 1 0 0 0 0 1 0 1 0 0 0 0 1 v9
bart vec 0 1 0 0 0 0 1 0 1 0 0 0 0 1 va
bart vec 0 0 0 0 0 0 0 0 0 0 0 0 0 0 vb
bart vec 0 0 0 1 1 1 0 0 0 1 1 1 0 0 vc
bart join 1 v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 va vb vc v


bart resize -c 0 50 1 50 v o

bart conway -n3 o x


# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>

macro(setup_linalg_vendor_on_success vendor target)
  if(${target}_FOUND)
    set(LINALG_VENDOR_LIB ${ARGN})
    set(LINALG_VENDOR_FOUND TRUE)
    set(LINALG_VENDOR ${vendor} CACHE STRING "Vendor type for linear algebra library" FORCE)
    set(LINALG_VENDOR_TGT ${target})
  endif()
endmacro()

##==============================================================

if(NOT "${LINALG_VENDOR}" STREQUAL "")
  # If the user specifies LINALG_VENDOR, make sure we find what he wants
  set(FIND_PACKAGE_ARG "REQUIRED")
else()
  set(FIND_PACKAGE_ARG)
endif()

## Switch based on the linear algebra optimized library to 
## use.  Note that the first library successfully found
## will be used 
##
## -*-*- Try with only a BLAS library first if BART_NO_LAPACKE is set
if(BART_NO_LAPACKE AND (NOT LINALG_VENDOR OR LINALG_VENDOR STREQUAL "BLAS"))
  find_package(LAPACKE ${FIND_PACKAGE_ARG} COMPONENTS lapack cblas blas)
  setup_linalg_vendor_on_success("BLAS" LAPACKE LAPACKE::LAPACK LAPACKE::CBLAS LAPACKE::BLAS)
endif()

## -*-*- Try AMD BLIS/libFlame next
if(NOT LINALG_VENDOR OR LINALG_VENDOR MATCHES "AMD_FLAME")
  find_package(libFlame ${FIND_PACKAGE_ARG})
  setup_linalg_vendor_on_success("AMD_FLAME" libFlame libFlame::libFlame)

  if(libFlame_FOUND)
    if(NOT BART_NO_LAPACKE)
      find_package(LAPACKE REQUIRED COMPONENTS LAPACKE)
      get_target_property(
	libFlame_INTERFACE_LINK_LIBRARIES
	${LINALG_VENDOR_LIB}
	INTERFACE_LINK_LIBRARIES)
      set_target_properties(${LINALG_VENDOR_LIB} PROPERTIES
	INTERFACE_LINK_LIBRARIES "${libFlame_INTERFACE_LINK_LIBRARIES};LAPACKE::LAPACKE")
    endif()
  endif()
endif()

## -*-*- Try OpenBLAS next
if(NOT LINALG_VENDOR OR LINALG_VENDOR MATCHES "OpenBLAS")
  find_package(OpenBLAS ${FIND_PACKAGE_ARG})
  setup_linalg_vendor_on_success("OpenBLAS" OpenBLAS OpenBLAS::OpenBLAS)
endif()

##
## -*-*- Try ATLAS version next
if(NOT LINALG_VENDOR OR LINALG_VENDOR MATCHES "ATLAS")
  find_package(ATLAS ${FIND_PACKAGE_ARG})
  setup_linalg_vendor_on_success("ATLAS" ATLAS ATLAS::ATLAS)
endif()

##
## -*-*- Try Generic LAPACKE version Last
if(NOT LINALG_VENDOR OR LINALG_VENDOR MATCHES "LAPACKE")
  #NOTE: By specifying Fortran here, linking to lapack becomes easier
  # See https://blog.kitware.com/fortran-for-cc-developers-made-easier-with-cmake/
  if(NOT WIN32)
    enable_language(Fortran)
  endif()
  ## Only very new versions of LAPACK (> 3.5.0) have built in support
  ## for cblas and lapacke.  This method is not very robust to older
  ## versions of lapack that might be able to be supported.
  ## It is know to work local builds
  find_package(LAPACKE ${FIND_PACKAGE_ARG})
  setup_linalg_vendor_on_success("LAPACKE" LAPACKE LAPACKE::LAPACKE LAPACKE::LAPACKE LAPACKE::CBLAS LAPACKE::BLAS)
endif()

# Fix weird error when compiling on Mac...
if(APPLE)
  include_directories(${${LINALG_VENDOR_TGT}_INCLUDE_DIRS})
endif()

##
## -*-*- Finally, set include_directories -*-*-*
if(NOT LINALG_VENDOR_FOUND)
   message(FATAL_ERROR "No valid linear algebra libraries found!")
endif()

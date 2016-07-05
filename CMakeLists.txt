# Copyright 2015-2016. Hans J. Johnson <hans-johnson@uiowa.edu>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Hans J. Johnson <hans-johnson@uiowa.edu>

cmake_minimum_required(VERSION 2.8.7)
project(bart C)

enable_language(C)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

## Options
##- TODO option(USE_CUDA    "Provide support for CUDA processing" OFF)
##- TODO option(USE_ACML    "Provide support for ACML processing" OFF)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
##- TODO option(USE_SLINK "Provide SLINK support" OFF)
# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()
option(USE_ISMRMRD "Use external ISMRMRD package for reading/writing" OFF)

## Compiler flags -- TODO This could be better, see ITK it won't work on windows builds
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -O3") # -Wall -Wextra -Wmissing-prototypes")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ffast-math -O3") # -Wall -Wextra")
endif()

find_package(CBLAS NO_MODULE REQUIRED)
message(STATUS "CBLAS_INCLUDE_DIRS: ${CBLAS_INCLUDE_DIRS}")
include_directories(${CBLAS_INCLUDE_DIRS})
find_package(LAPACKE NO_MODULE REQUIRED)
include_directories(${LAPACKE_INCLUDE_DIRS})
set(USE_FFTWF ON) # Only find single precision fftw
find_package(FFTW REQUIRED)
message(STATUS "FFTWF_LIBRARIES: ${FFTWF_LIBRARIES}")

find_package(PNG REQUIRED)
add_definitions(${PNG_DEFINITIONS})
include_directories(${PNG_INCLUDE_DIRS})


execute_process(COMMAND ${CMAKE_CURRENT_LIST_DIR}/git-version.sh OUTPUT_VARIABLE BART_VERSION_STRING WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
string(STRIP ${BART_VERSION_STRING} BART_VERSION_STRING) ## Remove trailing whitespace (return characters)
message(STATUS "BART VERSION: ${BART_VERSION_STRING}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/version.inc.in "VERSION(\@BART_VERSION_STRING\@)")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/version.inc.in ${CMAKE_CURRENT_BINARY_DIR}/version.inc @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(BART_SUPPORT_LIBS calib misc dfwavelet grecon iter linops lowrank noir noncart num sake sense simu wavelet2 wavelet3)
if(USE_ISMRMRD)
  list(APPEND BART_SUPPORT_LIBS ismrm)
endif()
include_directories(src)
set(bart_support_SRCS "")
foreach(curr_lib ${BART_SUPPORT_LIBS})
    file(GLOB ${curr_lib}_SRCS "src/${curr_lib}/*.c")
    list(APPEND bart_support_SRCS ${${curr_lib}_SRCS})
endforeach()

file(GLOB bartlib_SRCS src/*.c)
list(REMOVE_ITEM bartlib_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c) # Remove the main.c file from the glob function
## src/mat2cfl.c needs to be built special
list(REMOVE_ITEM bartlib_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat2cfl.c) # Remove the src/mat2cfl.c file from the glob function
list(APPEND bart_support_SRCS ${bartlib_SRCS})
add_library(bartsupport ${bart_support_SRCS})

set (ALLPROGS "")
set (TBASE show slice crop resize join transpose zeros ones flip circshift extract repmat bitmask reshape version)
list(APPEND ALLPROGS ${TBASE})
set(TFLP scale conj fmac saxpy sdot spow cpyphs creal carg normalize cdf97 pattern nrmse mip avg)
list(APPEND ALLPROGS ${TFLP})
set(TNUM fft fftmod fftshift noise bench threshold conv rss filter)
list(APPEND ALLPROGS ${TNUM})
set(TRECO pics pocsense rsense sqpics bpsense itsense nlinv nufft rof sake wave lrmatrix estdims)
list(APPEND ALLPROGS ${TRECO})
set(TCALIB ecalib ecaltwo caldir walsh cc calmat svd estvar)
list(APPEND ALLPROGS ${TCALIB})
set(TMRI homodyne poisson twixread fakeksp)
list(APPEND ALLPROGS ${TMRI})
set(TSIM phantom traj)
list(APPEND ALLPROGS ${TSIM})
set(TIO toimg)
list(APPEND ALLPROGS ${TIO})

set(FORTRAN_SUPPORT_LIB gfortran)

file(READ src/main.c MAIN_TEMPLATE_STRING)
foreach(curr_prog ${ALLPROGS})
  set(main_real ${curr_prog})
  string(REPLACE "main_real" "main_${curr_prog}" CURR_PROG_STRING "${MAIN_TEMPLATE_STRING}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${curr_prog}.c "${CURR_PROG_STRING}")
  add_executable(${curr_prog} ${CMAKE_CURRENT_BINARY_DIR}/${curr_prog}.c)
  target_link_libraries(${curr_prog} bartsupport ${PNG_LIBRARIES} ${FFTWF_LIBRARIES} ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES} ${BLAS_LIBRARIES} ${FORTRAN_SUPPORT_LIB} )
endforeach()



include_directories(${CMAKE_CURRENT_LIST_DIR}/utests)
set(UNIT_TEST_SRC utests/test_batchsvd.c  utests/test_flpmath.c   utests/test_pattern.c   utests/test_splines.c)

set(UTESTS
"call_test_batch_svthresh_tall,
call_test_batch_svthresh_wide,
")
file(READ utests/utest.c TEST_DRIVER_TEMPLATE)
string(REPLACE "UTESTS" "${UTESTS}" TEST_DRIVER_CODE "${TEST_DRIVER_TEMPLATE}")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/utest.c "${TEST_DRIVER_CODE}")
#configure_file(utests/utest.c.in ${CMAKE_CURRENT_BINARY_DIR}/utest.c @ONLY)
add_executable(utest ${CMAKE_CURRENT_BINARY_DIR}/utest.c ${UNIT_TEST_SRC})
target_link_libraries(utest bartsupport ${PNG_LIBRARIES} ${FFTWF_LIBRARIES} ${LAPACKE_LIBRARIES} ${LAPACK_LIBRARIES} ${CBLAS_LIBRARIES} ${BLAS_LIBRARIES} ${FORTRAN_SUPPORT_LIB} )

#-----------------------------------------------------------------------------
include(CTest)
enable_testing()
add_test(NAME BartUTest COMMAND $<TARGET_FILE:utest>)


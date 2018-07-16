# Copyright 2015-2016. Hans J. Johnson <hans-johnson@uiowa.edu>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Hans J. Johnson <hans-johnson@uiowa.edu>
cmake_minimum_required(VERSION 2.8.7)

project(bart C)
enable_language(C)
enable_language(CXX)


# http://stackoverflow.com/questions/24840030/forcing-c99-in-cmake-to-use-for-loop-initial-declaration
macro(use_c99)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_C_FLAGS "--std=gnu11 ${CMAKE_C_FLAGS}")
    endif ()
  else ()
    set (CMAKE_C_STANDARD 99)
  endif ()
endmacro(use_c99)
use_c99()

# ==============================================================================

option(USE_LOG_BACKEND "Enable delegating all outputs to external logging backend" OFF)

include(CMakeDependentOption)
cmake_dependent_option(USE_LOG_SIEMENS_BACKEND "Use the Siemens logging backend" OFF
  "USE_LOG_BACKEND" OFF)
if(USE_LOG_SIEMENS_BACKEND)
  message(STATUS "Delegating all outputs to the Siemens logging backend")
endif(USE_LOG_SIEMENS_BACKEND)

cmake_dependent_option(USE_LOG_ORCHESTRA_BACKEND "Use the Orchestra logging backend" OFF
  "USE_LOG_BACKEND" OFF)
if(USE_LOG_ORCHESTRA_BACKEND)
  message(STATUS "Delegating all outputs to the Orchestra logging backend")
endif(USE_LOG_ORCHESTRA_BACKEND)

cmake_dependent_option(USE_LOG_GADGETRON_BACKEND "Use the Gadgetron logging backend" OFF
  "USE_LOG_BACKEND" OFF)
if(USE_LOG_GADGETRON_BACKEND)
  message(STATUS "Delegating all outputs to the Gadgetron logging backend")
endif(USE_LOG_GADGETRON_BACKEND)

# ==============================================================================

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake ${CMAKE_MODULE_PATH})

# --- Provide good defaults for searching for packages (i.e. ismrmrd)
set(CMAKE_PREFIX_PATH "")
if(CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "/usr/local")
endif()
if(EXISTS $ENV{CMAKE_PREFIX_PATH})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif()
if(EXISTS $ENV{ISMRMRD_HOME})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ISMRMRD_HOME})
endif()
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
## -message(STATUS "Looking for packages in : ${CMAKE_PREFIX_PATH}")

## Options
option(USE_CUDA    "Provide support for CUDA processing" OFF)
if(USE_CUDA)
  enable_language(CXX)
  find_package(CUDA)
  add_definitions(-DUSE_CUDA)
  CUDA_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_LIST_DIR}/src)
  # set(CUDA_NVCC_FLAGS "-DUSE_CUDA;-Xcompiler;-fPIC;-Xcompiler;-fopenmp;-O3;-arch=sm_20;-m64;-ccbin ${CMAKE_C_COMPILER}")
  set(CUDA_NVCC_FLAGS "-DUSE_CUDA;-Xcompiler;-fPIC;-Xcompiler;-fopenmp;-O3;-arch=sm_20;-ccbin ${CMAKE_C_COMPILER}")
  macro(bart_add_executable target_name)
    CUDA_ADD_EXECUTABLE(${target_name} ${ARGN})
    CUDA_ADD_CUFFT_TO_TARGET(${target_name})
    CUDA_ADD_CUBLAS_TO_TARGET(${target_name})
    target_link_libraries(${target_name} ${CUDA_LIBRARIES})
  endmacro(bart_add_executable)
  macro(bart_add_library target_name)
    CUDA_ADD_LIBRARY(${target_name} ${ARGN} STATIC)
    CUDA_ADD_CUFFT_TO_TARGET(${target_name})
    CUDA_ADD_CUBLAS_TO_TARGET(${target_name})
    target_link_libraries(${target_name} ${CUDA_LIBRARIES})
  endmacro(bart_add_library)  
else()
  macro(bart_add_executable target_name)
    add_executable(${target_name} ${ARGN})
  endmacro(bart_add_executable)
  macro(bart_add_library target_name)
    add_library(${target_name} ${ARGN})
  endmacro(bart_add_library)
endif(USE_CUDA)

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

find_package(ISMRMRD QUIET) ## if you can find ISMRMRD by default, then default configuration is ON
option(USE_ISMRMRD "Use external ISMRMRD package for reading/writing" ${ISMRMRD_FOUND})
if(USE_ISMRMRD)
  find_package(ISMRMRD REQUIRED)
endif()

## Compiler flags -- TODO This could be better, see ITK it won't work on windows builds
if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -O3") # -Wall -Wextra -Wmissing-prototypes")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ffast-math -O3") # -Wall -Wextra")
endif()

##==============================================================
## Switch based on the linear algebra optimized library to 
## use.  Note that the first library successfully found
## will be used 
##
## -*-*- Try OpenBLAS first
if(NOT LINALG_VENDOR OR LINALG_VENDOR MATCHES "OpenBLAS")
  include( ${CMAKE_CURRENT_LIST_DIR}/cmake/FindOpenBLAS.cmake )
  if(OpenBLAS_FOUND)
     set(LINALG_VENDOR_FOUND TRUE)
     set(LINALG_VENDOR "OpenBLAS")
     set(LINALG_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS})
     if(OpenBLAS_HAS_PARALLEL_LIBRARIES)
       # HACK!! set(LINALG_LIBRARIES ${OpenBLAS_PARALLEL_LIBRARIES})
        set(LINALG_LIBRARIES ${OpenBLAS_LIBRARIES})
     else()
        set(LINALG_LIBRARIES ${OpenBLAS_LIBRARIES})
     endif()
  endif()
endif()

##
## -*-*- Try ATLAS version next
if(NOT LINALG_VENDOR OR LINALG_VENDOR MATCHES "ATLAS")
  include( ${CMAKE_CURRENT_LIST_DIR}/cmake/FindATLAS.cmake )
  if(ATLAS_FOUND)
     set(LINALG_VENDOR_FOUND TRUE)
     set(LINALG_VENDOR "ATLAS")
     set(LINALG_INCLUDE_DIRS ${ATLAS_INCLUDE_DIRS})
     set(LINALG_LIBRARIES ${ATLAS_LIBRARIES})
  endif()
endif()

##
## -*-*- Try Generic LAPACKE version Last
if(NOT LINALG_VENDOR OR LINALG_VENDOR MATCHES "LAPACKE")
  #NOTE: By specifying Fortran here, linking to lapack becomes easier
  # See https://blog.kitware.com/fortran-for-cc-developers-made-easier-with-cmake/  
  enable_language(Fortran)
  ## Only very new versions of LAPACK (> 3.5.0) have built in support
  ## for cblas and lapacke.  This method is not very robust to older
  ## versions of lapack that might be able to be supported.
  ## It is know to work local builds
  find_package(LAPACKE REQUIRED)
  if(LAPACKE_FOUND)
     set(LINALG_VENDOR_FOUND TRUE)
     set(LINALG_VENDOR "LAPACKE")
     set(LINALG_INCLUDE_DIRS ${LAPACKE_INCLUDE_DIRS})
     set(LINALG_LIBRARIES ${LAPACKE_LIBRARIES})
   endif()
endif()

##
## -*-*- Finally, set include_directories -*-*-*
if(NOT LINALG_VENDOR_FOUND)
   message(FATAL_ERROR "No valid linear algebra libraries found!")
endif()
include_directories(${LINALG_INCLUDE_DIRS})

##======================================================================
set(USE_FFTWF ON) # Only find single precision fftw
find_package(FFTW REQUIRED)
message(STATUS "FFTWF_LIBRARIES: ${FFTWF_LIBRARIES}")

find_package(PNG REQUIRED)
add_definitions(${PNG_DEFINITIONS})
include_directories(${PNG_INCLUDE_DIRS})

execute_process(COMMAND ${CMAKE_CURRENT_LIST_DIR}/git-version.sh OUTPUT_VARIABLE BART_FULL_VERSION_STRING WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
string(STRIP ${BART_FULL_VERSION_STRING} BART_FULL_VERSION_STRING) ## Remove trailing whitespace (return characters)
string(REGEX REPLACE ".*v([0-9]*)\\.([0-9]*)\\.([0-9]*)-.*" "\\1" BART_VERSION_MAJOR "${BART_FULL_VERSION_STRING}")
string(REGEX REPLACE ".*v([0-9]*)\\.([0-9]*)\\.([0-9]*)-.*" "\\2" BART_VERSION_MINOR "${BART_FULL_VERSION_STRING}")
string(REGEX REPLACE ".*v([0-9]*)\\.([0-9]*)\\.([0-9]*)-.*" "\\3" BART_VERSION_PATCH "${BART_FULL_VERSION_STRING}")
message(STATUS "BART VERSION: ${BART_FULL_VERSION_STRING}:  ${BART_VERSION_MAJOR} ${BART_VERSION_MINOR} ${BART_VERSION_PATCH}")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/version.inc.in "VERSION(\@BART_FULL_VERSION_STRING\@)")
configure_file(${CMAKE_CURRENT_BINARY_DIR}/version.inc.in ${CMAKE_CURRENT_BINARY_DIR}/version.inc @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

## ========== Process build_targets.mk
set (ALLPROGS "")
file(READ "${CMAKE_CURRENT_LIST_DIR}/build_targets.mk" bld_tgt_list_of_list)
# Convert file contents into a CMake list (where each element in the list
# is one line of the file)
#
string(REGEX REPLACE ";" "\\\\;" bld_tgt_list_of_list "${bld_tgt_list_of_list}")
string(REGEX REPLACE "\n" ";" bld_tgt_list_of_list "${bld_tgt_list_of_list}")
foreach(bld_tgt_line ${bld_tgt_list_of_list})
   if( "${bld_tgt_line}" MATCHES "^[^#].*=.*")
     string(REGEX REPLACE "^ *([^=]*) *= *(.*) *" "\\1" BLD_KEY "${bld_tgt_line}")
     string(REGEX REPLACE "^ *([^=]*) *= *(.*) *" "\\2" BLD_VALUE "${bld_tgt_line}")
     string(REPLACE " " ";" ${BLD_KEY} "${BLD_VALUE}")  ## Create a new variable called ${BLD_KEY}
     ## message(STATUS "KEY:${BLD_KEY}:    VALUE:${${BLD_KEY}}:")
     list(APPEND ALLPROGS ${${BLD_KEY}})
   endif()
endforeach()

## BART Makefile depends heavily on pre-processor replacements
## on the command line rather than a library of common functionality
## that is used to compile code once and link many times.
## In this cmake build, it was choosen to generate separate files
## for separate build uses of each source file.
## This convoluted mechansims is needed to avoid changing
## any build organizaiton so that the Makefile build system is
## not changed at all.  A different organziation of the build
## proceedures could make this much less complicated.
##
## usage CONFIG_BY_REPLACEMENT( in.file out.file "at top string" match1 rep1 match2 rep2 .... matchN repN)
macro(CONFIG_BY_REPLACEMENT INFILE OUTFILE PREFIX )
  if(NOT EXISTS "${INFILE}")
    message(FATAL_ERROR "\n\nMISSING INPUT FILE \"${INFILE}\" for generating \"${OUTFILE}\"\n\n")
  endif()
  file(READ "${INFILE}" MAIN_TEMPLATE_STRING)

  set(all_repacement_pairs ${ARGN})
  list(LENGTH all_repacement_pairs RP_LENGTH)
  while(NOT ${RP_LENGTH} LESS 2)
    list(GET all_repacement_pairs 0 INSTRING)
    list(REMOVE_AT all_repacement_pairs 0 )
    list(LENGTH all_repacement_pairs RP_LENGTH)
    if(${RP_LENGTH} GREATER 0)
      list(GET all_repacement_pairs 0 OUTSTRING)
      list(REMOVE_AT all_repacement_pairs 0 )
      list(LENGTH all_repacement_pairs RP_LENGTH)
    else()
      message(FATAL_ERROR "Replacement pairs unmatched: ${ARGN}")
    endif()
  
    string(REPLACE "${INSTRING}" "${OUTSTRING}" MAIN_TEMPLATE_STRING "${MAIN_TEMPLATE_STRING}")
  endwhile()
  string(REPLACE "__EOL__" ";" MAIN_TEMPLATE_STRING "${MAIN_TEMPLATE_STRING}") ## UglyFix
  set(MAIN_TEMPLATE_STRING "${PREFIX}\n${MAIN_TEMPLATE_STRING}")
  
  if(EXISTS "${OUTFILE}" )
    file(READ "${OUTFILE}" PREVIOUS_CONTENT)
  else()
    set(PREVIOUS_CONTENT "")
  endif()
  string( COMPARE EQUAL "${PREVIOUS_CONTENT}" "${MAIN_TEMPLATE_STRING}" STRING_NO_CHANGE)
  if( NOT STRING_NO_CHANGE )
    file(WRITE "${OUTFILE}" "${MAIN_TEMPLATE_STRING}")
  endif()
endmacro()

set(BART_SUPPORT_LIBS calib misc dfwavelet grecon iter linops lowrank nlops noir noncart num sake sense simu wavelet)
if(USE_ISMRMRD)
  list(APPEND BART_SUPPORT_LIBS ismrm)
  link_directories(${ISMRMRD_LIBRARY_DIRS})
  include_directories(${ISMRMRD_INCLUDE_DIRS})
endif()
include_directories(src)
set(bart_support_SRCS "")
set(bart_support_CCSRCS "")
set(bart_support_CUSRCS "")
foreach(curr_lib ${BART_SUPPORT_LIBS})
    file(GLOB ${curr_lib}_SRCS "src/${curr_lib}/*.c")
    list(APPEND bart_support_SRCS ${${curr_lib}_SRCS})
  
    file(GLOB ${curr_lib}_CCSRCS "src/${curr_lib}/*.cc")
    list(APPEND bart_support_SRCS ${${curr_lib}_CCSRCS})

    if(USE_CUDA)
      file(GLOB ${curr_lib}_CUSRCS "src/${curr_lib}/*.cu")
      list(APPEND bart_support_CUSRCS ${${curr_lib}_CUSRCS})
    endif(USE_CUDA)
endforeach()
  
bart_add_library(bartsupport ${bart_support_SRCS})

set(LOG_BACKEND_SRCS "")
list(REMOVE_ITEM bart_support_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/UTrace.cc")
list(REMOVE_ITEM bart_support_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/Orchestra.cc")
list(REMOVE_ITEM bart_support_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/Gadgetron.cc")
if(USE_LOG_BACKEND)
  if(USE_LOG_SIEMENS_BACKEND)
    # No need to add any include paths as the required code was copied into UTrace.cc from Siemens' IDEA framework
    list(APPEND LOG_BACKEND_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/UTrace.cc")
  endif(USE_LOG_SIEMENS_BACKEND)
  if(USE_LOG_ORCHESTRA_BACKEND)
    set(OX_INSTALL_DIRECTORY "" CACHE PATH "Install directory of the GE Orchestra SDK")
    set(TOPDIR "${OX_INSTALL_DIRECTORY}/include")
    
    # Include recon libraries exported from the SDK CMake build
    set(RECON_LIBRARIES_INCLUDE_FILE ${OX_INSTALL_DIRECTORY}/lib/ReconLibraries.cmake)
    if(EXISTS ${RECON_LIBRARIES_INCLUDE_FILE})
      include (${RECON_LIBRARIES_INCLUDE_FILE})
    else(EXISTS ${RECON_LIBRARIES_INCLUDE_FILE})
      message("Could not find ${RECON_LIBRARIES_INCLUDE_FILE}")
      message(FATAL_ERROR "Verify that the CMake OX_INSTALL_DIRECTORY option is set correctly")
    endif(EXISTS ${RECON_LIBRARIES_INCLUDE_FILE})
    
    # Include Orchestra SDK build configuration
    include (${TOPDIR}/recon/SDK/product.cmake)

    # FIXME: need to check which libraries are actually required
    target_link_libraries(bartsupport Core Common Control System ${OX_3P_LIBS} ${OX_OS_LIBS})
    
    list(APPEND LOG_BACKEND_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/Orchestra.cc")
  endif(USE_LOG_ORCHESTRA_BACKEND)
  if(USE_LOG_GADGETRON_BACKEND)
    if(NOT DEFINED GADGETRON_CMAKE_MODULES)
      find_path(GADGETRON_CMAKE_MODULES FindGadgetron.cmake
	HINTS
	$ENV{GADGETRON_HOME}/share/gadgetron/cmake
	/usr/local/gadgetron)

      if(NOT GADGETRON_CMAKE_MODULES)
	message(FATAL_ERROR "GADGETRON_CMAKE_MODULES cannot be found. Try to set GADGETRON_HOME environment variable.")
      endif(NOT GADGETRON_CMAKE_MODULES)

      list(APPEND CMAKE_MODULE_PATH ${GADGETRON_CMAKE_MODULES})
    endif(NOT DEFINED GADGETRON_CMAKE_MODULES)

    find_package(Gadgetron REQUIRED)
    include_directories(${GADGETRON_HOME}/toolboxes/log)
    list(APPEND LOG_BACKEND_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/Gadgetron.cc")
  endif(USE_LOG_GADGETRON_BACKEND)
endif(USE_LOG_BACKEND)
set(LOG_BACKEND_SRCS "${LOG_BACKEND_SRCS}" CACHE INTERNAL "")
list(REMOVE_ITEM bart_support_SRCS "${LOG_BACKEND_SRCS}")

target_link_libraries(bartsupport ${PNG_LIBRARIES} ${FFTWF_LIBRARIES} ${LINALG_LIBRARIES})
if(USE_ISMRMRD)
  target_link_libraries(bartsupport ${ISMRMRD_LIBRARIES})
endif()
install(TARGETS bartsupport
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)


### PASS #1:  Build stand-alone programs
### Generate stand alone programs
#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/StandAloneCode)
#foreach(curr_prog ${ALLPROGS})
#  CONFIG_BY_REPLACEMENT( ${CMAKE_CURRENT_LIST_DIR}/src/main.c ${CMAKE_CURRENT_BINARY_DIR}/StandAloneCode/${curr_prog}.c
#                         "/* Generated by cmake */"
#                        "main_real" "main_${curr_prog}")
#  bart_add_executable(${curr_prog} ${CMAKE_CURRENT_BINARY_DIR}/StandAloneCode/${curr_prog}.c ${CMAKE_CURRENT_LIST_DIR}/src/${curr_prog}.c)
#  target_link_libraries(${curr_prog} bartsupport )
#  install(TARGETS ${curr_prog}
#        RUNTIME DESTINATION bin
#        LIBRARY DESTINATION lib
#        ARCHIVE DESTINATION lib/static)
#endforeach()

## More crafty file manipulation so that we maintain backward comaptibility
## with the Makefile

## Generate combined programs
#==============================================
set(EXTERN_LIST "\n\n/* Generated by cmake */\n")
foreach(driver ${ALLPROGS})
   set(EXTERN_LIST "${EXTERN_LIST}extern int main_${driver}(int argc, char* argv[])__EOL__\n")
endforeach()
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CombinedCode)
## Configure include header
CONFIG_BY_REPLACEMENT( "${CMAKE_CURRENT_LIST_DIR}/src/main.h" "${CMAKE_CURRENT_BINARY_DIR}/CombinedCode/config_main.h"
                       "/* Generated by cmake */"
                       "MAP(DECLMAIN, MAIN_LIST)" "${EXTERN_LIST}"
                       "misc/cppmap.h" "stdio.h")             ## Replacement 3 no longer need this include 
include_directories(${CMAKE_CURRENT_BINARY_DIR}/CombinedCode)

#==============================================
set(DRIVER_LIST "\n\n/* Generated by cmake */\n")
foreach(driver ${ALLPROGS})
   set(DRIVER_LIST "${DRIVER_LIST}{ main_${driver}, \"${driver}\" },\n")
endforeach()
set(ALL_BART_SRCS "")
foreach(curr_prog ${ALLPROGS})
  CONFIG_BY_REPLACEMENT( "${CMAKE_CURRENT_LIST_DIR}/src/${curr_prog}.c" "${CMAKE_CURRENT_BINARY_DIR}/CombinedCode/${curr_prog}.c"
                         "/* Generated by cmake */\n#include \"config_main.h\""
                         "MAP(DENTRY, MAIN_LIST)" "${DRIVER_LIST}")
  list(APPEND ALL_BART_SRCS "${CMAKE_CURRENT_BINARY_DIR}/CombinedCode/${curr_prog}.c")
endforeach()
foreach(curr_prog bart)
  CONFIG_BY_REPLACEMENT( "${CMAKE_CURRENT_LIST_DIR}/src/${curr_prog}.c" "${CMAKE_CURRENT_BINARY_DIR}/CombinedCode/${curr_prog}.c"
                         "#include \"config_main.h\""
                         "MAP(DENTRY, MAIN_LIST)" "${DRIVER_LIST}" ## Replacement 1
                         "main_bart" "main"                        ## Replacement 2
                         "misc/cppmap.h" "stdio.h")                ## Replacement 3 no longer need this include 
  list(APPEND ALL_BART_SRCS "${CMAKE_CURRENT_BINARY_DIR}/CombinedCode/${curr_prog}.c")
endforeach()

bart_add_executable(bart ${ALL_BART_SRCS} ${LOG_BACKEND_SRCS})
target_link_libraries(bart bartsupport)
install(TARGETS bart
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)


#==============================================
# TODO: Matlab code
option(USE_MATLAB "Specify if the optional matlab programs should be built" OFF)
if(USE_MATLAB)
  find_package(Matlab REQUIRED)
  if(MATLAB_FOUND)
    message(STATUS "MATLAB LIBRARIES FOUND: ${MATLAB_LIBRARIES_DIR}")
    include_directories(${MATLAB_INCLUDE_DIR})
    bart_add_executable(mat2cfl ${CMAKE_CURRENT_LIST_DIR}/src/mat2cfl.c ${LOG_BACKEND_SRCS})
    target_link_libraries(mat2cfl bartsupport ${MATLAB_MAT_LIBRARY} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY})

    install(TARGETS mat2cfl
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
  endif()
endif()

#==============================================
# Testing code
include(CTest)
enable_testing()

if(BUILD_TESTING)
  file(GLOB utests_list_SRCS "${CMAKE_CURRENT_LIST_DIR}/utests/test_*.c")
  foreach(f ${utests_list_SRCS})
    file(STRINGS ${f} ut_register_lines
      REGEX "UT_REGISTER_TEST\\(.*\\)")
    string(REGEX
      REPLACE "UT_REGISTER_TEST\\(([a-zA-Z0-9_ ]+)\\)[\\r\\n\\t ]*\\;" "\\1"
      tests_list
      "${ut_register_lines}")

    get_filename_component(test_name ${f} NAME_WE)
    set(${test_name}_tests "")
    foreach(sub_test_name ${tests_list})
      set(${test_name}_tests "${${test_name}_tests}call_${sub_test_name},")
    endforeach()

    add_executable(${test_name} ${CMAKE_CURRENT_LIST_DIR}/utests/utest.c ${f})
    target_link_libraries(${test_name} bartsupport)
    target_compile_definitions(${test_name} PUBLIC UTESTS=${${test_name}_tests})
    add_test(${test_name} ${test_name})
  endforeach()
endif(BUILD_TESTING)

# ==============================================================================

set(TARBALL_VERSION "${BART_VERSION_MAJOR}.${BART_VERSION_MINOR}.${BART_VERSION_PATCH}")
add_custom_target(tarball
    COMMAND git archive --prefix=bart-${TARBALL_VERSION}/ -o ${CMAKE_CURRENT_BINARY_DIR}/bart-${TARBALL_VERSION}.tar.gz v${TARBALL_VERSION}
    WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    COMMENT "BUILD TARBALL FOR BART WITH LATEST VERSION"
)

add_custom_target(doxygen
         COMMAND ${CMAKE_CURRENT_LIST_DIR}/makedoc.sh
         WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
         DEPENDS bart
         SOURCES makedoc.sh doxyconfig)

add_custom_target(bart.syms
         COMMAND ${CMAKE_CURRENT_LIST_DIR}/rules/make_symbol_table.sh $<TARGET_FILE:bart> ${CMAKE_CURRENT_BINARY_DIR}/bart.syms
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
         DEPENDS bart
         SOURCES makedoc.sh doxyconfig)

add_custom_command(TARGET bart
        POST_BUILD
        COMMAND ${CMAKE_CURRENT_LIST_DIR}/rules/update_commands.sh $<TARGET_FILE:bart> ${CMAKE_CURRENT_LIST_DIR}/doc/commands.txt ${ALLPROGS}
)

file(GLOB DOCS "${CMAKE_CURRENT_LIST_DIR}/doc/*.txt")
list(APPEND ${CMAKE_CURRENT_LIST_DIR}/README)
install( FILES ${DOCS} DESTINATION share/doc/bart/ )

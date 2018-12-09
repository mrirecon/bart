# Copyright 2015-2016. Hans J. Johnson <hans-johnson@uiowa.edu>
# Copyright 2017-2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Hans J. Johnson <hans-johnson@uiowa.edu>
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>
cmake_minimum_required(VERSION 2.8.12)

if(CMAKE_VERSION VERSION_GREATER 3.0)
  set(_policy_list CMP0012 CMP0028 CMP0042 CMP0048 CMP0051 CMP0054 CMP0057 CMP0074)
  foreach(_policy ${_policy_list})
    if(POLICY ${_policy})
      cmake_policy(SET ${_policy} NEW)
    endif()
  endforeach()
  # CMP0012: if() recognizes numbers and booleans
  # CMP0028: :: in target names
  # CMP0042: MACOS_RPATH
  # CMP0048: allow VERSION in project()
  # CMP0051: list TARGET_OBJECTS in SOURCES property
  # CMP0054: no more de-referencing of "expr" in if() statements
  # CMP0057: if IN_LIST
  # CMP0074: XXX_ROOT variables for find_package(XXX)
endif()
if(CMAKE_VERSION VERSION_LESS 3.10) # 3.10 instead of 3.3 due to CUDA support
  if(CMAKE_VERSION VERSION_GREATER 3.8.99)
    message(STATUS "From the BART maintainers: ignore the message below")
  endif()
  cmake_policy(SET CMP0023 OLD) # target_link_libraries mixed signatures
else()
  cmake_policy(SET CMP0023 NEW) # target_link_libraries mixed signatures
endif()

# ==============================================================================
# Internal helper variables

set(_compile_flags_c)
set(_compile_flags_cxx)
set(_compile_flags_c_release)
set(_compile_flags_cxx_release)

# ==============================================================================
# Macro definitions

include(${CMAKE_CURRENT_LIST_DIR}/cmake/bart_macros.cmake)

# ==============================================================================
# Get the version now, so that we can set it when creating the project

if(EXISTS ${CMAKE_CURRENT_LIST_DIR}/.git)
  find_package(Git QUIET)
  if(Git_FOUND)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} describe --match "v*" --dirty
      OUTPUT_VARIABLE _full_version
      RESULT_VARIABLE _ret
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})

    if(_ret EQUAL 0)
      string(STRIP ${_full_version} _full_version)
      # _full_version is of the form: v0.4.03-109-gf9c0f56-dirty
      string(REGEX MATCHALL "[0-9]+" _version_list ${_full_version})
      list(GET _version_list 0 _version_major)
      list(GET _version_list 1 _version_minor)
      list(GET _version_list 2 _version_patch)
      file(WRITE ${CMAKE_CURRENT_LIST_DIR}/version.txt
	"v${_version_major}.${_version_minor}.${_version_patch}\n")
    else()
      get_version_from_file(${CMAKE_CURRENT_LIST_DIR}/version.txt)
    endif()
    
  else()
    get_version_from_file(${CMAKE_CURRENT_LIST_DIR}/version.txt)
  endif()
else()
  get_version_from_file(${CMAKE_CURRENT_LIST_DIR}/version.txt)
endif()

# ==============================================================================
# Create the BART project

if(CMAKE_VERSION VERSION_LESS 3.0)
  project(BART C)
  set(PROJECT_VERSION_MAJOR ${_version_major})
  set(PROJECT_VERSION_MINOR ${_version_minor})
  set(PROJECT_VERSION_PATCH ${_version_patch})
  set(${CMAKE_PROJECT_NAME}_VERSION_MAJOR ${_version_major})
  set(${CMAKE_PROJECT_NAME}_VERSION_MINOR ${_version_minor})
  set(${CMAKE_PROJECT_NAME}_VERSION_PATCH ${_version_patch})
else()
  project(BART
    LANGUAGES C
    VERSION "${_version_major}.${_version_minor}.${_version_patch}")
endif()

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE
    PROPERTY
    STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/cmake)

#==============================================================================
# Options

# NB: must be before compiler flags since we might enable CXX or CUDA
include(${CMAKE_CURRENT_LIST_DIR}/cmake/bart_options.cmake)

#==============================================================================
# Setup compiler flags

# http://stackoverflow.com/questions/24840030/forcing-c99-in-cmake-to-use-for-loop-initial-declaration
if (CMAKE_VERSION VERSION_LESS 3.1)
  if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR
      CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set (CMAKE_C_FLAGS "--std=gnu11 ${CMAKE_C_FLAGS}")
  endif()
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
      CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set (CMAKE_CXX_FLAGS "--std=c++14 ${CMAKE_CXX_FLAGS}")
  endif()
else()
  set(CMAKE_C_STANDARD 11)
  set(CMAKE_C_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# ==============================================================================

# --- Provide good defaults for searching for packages (i.e. ismrmrd)
if(NOT CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "/usr/local")
endif()
if(EXISTS $ENV{CMAKE_PREFIX_PATH})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif()
if(EXISTS $ENV{ISMRMRD_HOME})
  list(APPEND CMAKE_PREFIX_PATH $ENV{ISMRMRD_HOME})
endif()
if(CMAKE_PREFIX_PATH)
  list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
endif()

# --- Mac specific setup options
if(APPLE)
  # Look for MacPorts only as Homebrew users do not require any particular setup
  find_program(MACPORTS
    NAMES port
    PATHS /opt/local/bin /opt/local/sbin)
  if(MACPORTS)
    add_definitions(-DUSE_MACPORTS)
  endif()
endif()

# ==============================================================================
# Set compiler flags

check_for_compiler_flags(_compile_flags_release
  "-ffast-math /fp:fast -fast"
  "-O3 /Ox"
  )

if("${CMAKE_C_COMPILER_ID}" STREQUAL "Clang"
    OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND _compile_flags_c "-fblocks")
  list(APPEND _compile_flags_cxx "-fblocks")
endif()

if(CMAKE_VERSION VERSION_GREATER 3.2.99)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C>:${_compile_flags_c}>"
    "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:C>>:${_compile_flags_release_c}>"
    )
  if(CMAKE_CXX_COMPILER_LOADED)
    add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${_compile_flags_cxx}>"
    "$<$<AND:$<CONFIG:RELEASE>,$<COMPILE_LANGUAGE:CXX>>:${_compile_flags_release_cxx}>"
      )
  endif()
else()
  foreach(_flag ${_compile_flags_c})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_flag}")
  endforeach()
  foreach(_flag ${_compile_flags_c_release})
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${_flag}")
  endforeach()
  if(CMAKE_CXX_COMPILER_LOADED)
    foreach(_flag ${_compile_flags_cxx})
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_flag}")
    endforeach()
    foreach(_flag ${_compile_flags_cxx_release})
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${_flag}")
    endforeach()
  endif()
endif()

set(CMAKE_DEBUG_POSTFIX "d")
set(CMAKE_RELWITHDEBINFO_POSTFIX "rd")
set(CMAKE_MINSIZEREL_POSTFIX "rm")

# ==============================================================================
# Find a BLAS-capable library

include(${CMAKE_CURRENT_LIST_DIR}/cmake/BARTFindBLASlib.cmake)

##======================================================================

if(DEFINED _full_version)
  set(BART_FULL_VERSION_STRING ${_full_version})
else()
  set(BART_FULL_VERSION_STRING "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")
endif()
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

if(DEFINED BART_EXTRA_PROGS)
  list(APPEND ALLPROGS ${BART_EXTRA_PROGS})
endif()

if(BART_ISMRMRD)
  list(APPEND ALLPROGS ismrmrd) 
endif()

if(BART_DISABLE_PNG)
  list(REMOVE_ITEM ALLPROGS "toimg")
endif()

list(SORT ALLPROGS)


set(BART_SUPPORT_LIBS)
file(GLOB support_libs ${CMAKE_CURRENT_LIST_DIR}/src/*)
foreach(_f ${support_libs})
  if(IS_DIRECTORY ${_f})
    get_filename_component(_name ${_f} NAME)
    if(NOT "${_name}" STREQUAL "python")
      list(APPEND BART_SUPPORT_LIBS ${_name})
    endif()
  endif()
endforeach()

if(BART_ISMRMRD)
  link_directories(${ISMRMRD_LIBRARY_DIRS})
  include_directories(${ISMRMRD_INCLUDE_DIRS})
else()
    list(REMOVE_ITEM BART_SUPPORT_LIBS ismrm)
endif()

if(NOT BART_NO_LAPACKE)
  list(REMOVE_ITEM BART_SUPPORT_LIBS lapacke)
endif()

include_directories(${CMAKE_CURRENT_LIST_DIR}/src)

set(bart_support_SRCS "")
set(bart_support_CCSRCS "")
set(bart_support_CUSRCS "")
foreach(curr_lib ${BART_SUPPORT_LIBS})
    file(GLOB ${curr_lib}_SRCS "src/${curr_lib}/*.c")
    list(APPEND bart_support_SRCS ${${curr_lib}_SRCS})
  
    file(GLOB ${curr_lib}_CCSRCS "src/${curr_lib}/*.cc")
    list(APPEND bart_support_SRCS ${${curr_lib}_CCSRCS})

    file(GLOB ${curr_lib}_CUSRCS "src/${curr_lib}/*.cu")
    list(APPEND bart_support_CUSRCS ${${curr_lib}_CUSRCS})
endforeach()

# ------------------------------------------------------------------------------
# These files need to be compiled separately since they might contain
# Python code

set(BARTSUPPORT_PYTHON_SEPARATE_COMPILATION_FILES
  "${CMAKE_CURRENT_LIST_DIR}/src/misc/mmiocc.cc"
  "${CMAKE_CURRENT_LIST_DIR}/src/misc/mmio.c"
  "${CMAKE_CURRENT_LIST_DIR}/src/misc/misc.c")
foreach(file ${BARTSUPPORT_PYTHON_SEPARATE_COMPILATION_FILES})
  list(REMOVE_ITEM bart_support_SRCS "${file}")
endforeach()

if(NOT BART_ENABLE_MEM_CFL)
  list(REMOVE_ITEM
    BARTSUPPORT_PYTHON_SEPARATE_COMPILATION_FILES
    "${CMAKE_CURRENT_LIST_DIR}/src/misc/mmiocc.cc")
endif()

# ------------------------------------------------------------------------------

if(BART_DISABLE_PNG)
  list(REMOVE_ITEM bart_support_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/png.c")
endif()

set(LOG_BACKEND_SRCS "")
list(REMOVE_ITEM bart_support_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/UTrace.cc")
list(REMOVE_ITEM bart_support_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/Orchestra.cc")

if(BART_LOG_BACKEND)
  set(DEBUG_C_FILE "${CMAKE_CURRENT_LIST_DIR}/src/misc/debug.c")
  set(LOG_BACKEND_SRCS "${DEBUG_C_FILE}")
  
  if(BART_LOG_SIEMENS_BACKEND)
    if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/src/misc/UTrace.cc")
      set(_ERR "Could not find the src/misc/UTrace.cc file\n")
      set(_ERR "${_ERR}You can download a copy of UTrace.cc at:\n")
      set(_ERR "${_ERR}https://www.mr-idea.com/communities/idea/Lists/Customer/UTrace.cc\n")
      message(FATAL_ERROR "${_ERR}")
    endif()
    # No need to add any include paths as the required code was copied into UTrace.cc from Siemens' IDEA framework
    list(APPEND LOG_BACKEND_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/UTrace.cc")
  endif()

  # ------------------------------------
  
  if(BART_LOG_ORCHESTRA_BACKEND)
    set(OX_INSTALL_DIRECTORY "" CACHE PATH "Install directory of the GE Orchestra SDK")
    set(TOPDIR "${OX_INSTALL_DIRECTORY}/include")
    
    # Include recon libraries exported from the SDK CMake build
    set(RECON_LIBRARIES_INCLUDE_FILE ${OX_INSTALL_DIRECTORY}/lib/ReconLibraries.cmake)
    if(EXISTS ${RECON_LIBRARIES_INCLUDE_FILE})
      include (${RECON_LIBRARIES_INCLUDE_FILE})
    else(EXISTS ${RECON_LIBRARIES_INCLUDE_FILE})
      message("Could not find ${RECON_LIBRARIES_INCLUDE_FILE}")
      message(FATAL_ERROR "Verify that the CMake OX_INSTALL_DIRECTORY option is set correctly")
    endif()
    
    # Include Orchestra SDK build configuration
    include (${TOPDIR}/recon/SDK/product.cmake)

    list(APPEND LOG_BACKEND_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/misc/Orchestra.cc")
  endif()

  # ------------------------------------

  if(BART_LOG_GADGETRON_BACKEND)
    # Actually we do nothing here, Gagdetron will provide the vendor_log function
  endif()
endif()

foreach(file ${LOG_BACKEND_SRCS})
  list(REMOVE_ITEM bart_support_SRCS "${file}")
endforeach()

if(USE_CUDA)
  if("${CMAKE_VERSION}" VERSION_LESS 3.10)
    cuda_wrap_srcs(bartsupport_objs OBJ bartsupport_cuda_objs ${bart_support_CUSRCS})
  else()
    set(bartsupport_cuda_objs ${bart_support_CUSRCS})
  endif()
endif()

bart_add_object_library(bartsupport_objs ${bart_support_SRCS})
bart_add_library(bartsupport STATIC $<TARGET_OBJECTS:bartsupport_objs> ${DEBUG_C_FILE} ${BARTSUPPORT_PYTHON_SEPARATE_COMPILATION_FILES} ${bartsupport_cuda_objs})

target_link_libraries(bartsupport PUBLIC
  ${LINALG_VENDOR_LIB}
  ${PNG_TGT}
  ${FFTW_TGT}
  ${OpenMP_TGT})

if(BART_LOG_BACKEND)
  if(BART_LOG_ORCHESTRA_BACKEND)
    # FIXME: need to check which libraries are actually required
    target_link_libraries(bartsupport PUBLIC "Core;Common;Control;System;${OX_3P_LIBS};${OX_OS_LIBS}")    
  endif()
endif()

if(BART_ISMRMRD)
  target_link_libraries(bartsupport PUBLIC ${ISMRMRD_LIBRARIES})
endif()
set_target_properties(bartsupport
  PROPERTIES
  EXPORT_NAME BARTSUPPORT
  INTERFACE_INCLUDE_DIRECTORIES $<INSTALL_INTERFACE:include/bart>)

## Generate combined programs
#==============================================

set(MAIN_LIST)
foreach(prog ${ALLPROGS})
  set(MAIN_LIST "${MAIN_LIST} ${prog},")
endforeach()
set(MAIN_LIST "${MAIN_LIST} ()")
string(STRIP ${MAIN_LIST} MAIN_LIST)

#==============================================

set(ALL_BART_SRCS "")
foreach(curr_prog ${ALLPROGS})
  configure_file_add_header("${CMAKE_CURRENT_LIST_DIR}/src/${curr_prog}.c"
    "/* Generated by cmake */;#include \"main.h\"")
  list(APPEND ALL_BART_SRCS ${OUTFILE})
endforeach()
list(APPEND ALL_BART_SRCS "${CMAKE_CURRENT_LIST_DIR}/src/bart.c")

bart_add_executable(bart ${ALL_BART_SRCS} ${LOG_BACKEND_SRCS})
target_compile_definitions(bart PRIVATE -DMAIN_LIST=${MAIN_LIST} -Dmain_bart=main)
target_link_libraries(bart PRIVATE bartsupport)
set_target_properties(bart PROPERTIES EXPORT_NAME BART)
      
# ==============================================================================

bart_add_object_library(bartmain_objs ${ALL_BART_SRCS})
bart_add_library(bartmain
  STATIC
  $<TARGET_OBJECTS:bartsupport_objs>
  $<TARGET_OBJECTS:bartmain_objs>
  ${LOG_BACKEND_SRCS}
  ${BARTSUPPORT_PYTHON_SEPARATE_COMPILATION_FILES}
  ${bartsupport_cuda_objs})

target_compile_definitions(bartmain_objs PRIVATE -DMAIN_LIST=${MAIN_LIST})
target_link_libraries(bartmain PUBLIC
  ${LINALG_VENDOR_LIB} 
  ${PNG_TGT}
  ${FFTW_TGT}
  ${OpenMP_TGT}
  )

if(BART_ISMRMRD)
  target_link_libraries(bartmain PUBLIC ${ISMRMRD_LIBRARIES})
endif()

set_target_properties(bartmain
  PROPERTIES
  EXPORT_NAME BARTMAIN
  INTERFACE_INCLUDE_DIRECTORIES $<INSTALL_INTERFACE:include/bart>)

if(BART_ENABLE_MEM_CFL)
  set_target_properties(bartmain PROPERTIES LINKER_LANGUAGE CXX)
endif()

if(BART_LOG_BACKEND)
  # ensure that src/misc/misc.c will be compiled with logging backend support
  # but *ONLY* for the bartmain target
  target_compile_definitions(bartmain PRIVATE -DUSE_LOG_BACKEND)
endif()

# ==============================================================================

if(BART_CREATE_PYTHON_MODULE)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/bart_python.cmake)

  configure_file(${CMAKE_CURRENT_LIST_DIR}/src/python/pyBART.cpp.in ${PROJECT_BINARY_DIR}/pyBART.cpp @ONLY)

  bart_add_library(pyBART
    SHARED
    ${PROJECT_BINARY_DIR}/pyBART.cpp
    ${BARTSUPPORT_PYTHON_SEPARATE_COMPILATION_FILES}
    $<TARGET_OBJECTS:bartsupport_objs>
    $<TARGET_OBJECTS:bartmain_objs>)

  if(CMAKE_VERSION VERSION_LESS 3.1)
    target_include_directories(pyBART PRIVATE ${Python_INCLUDE_DIR})
  endif()
  target_compile_definitions(pyBART PRIVATE -DBART_WITH_PYTHON)
  target_link_libraries(pyBART PUBLIC "${Python_TGT};${PNG_TGT};${FFTW_TGT};${LINALG_VENDOR_LIB}")
  if(BART_ISMRMRD)
    target_link_libraries(pyBART PUBLIC ${ISMRMRD_LIBRARIES})
  endif()
  
  if(APPLE)
    set_target_properties(pyBART
      PROPERTIES
      SUFFIX ".so")
  endif()
  set_target_properties(pyBART
    PROPERTIES
    PREFIX ""
    DEBUG_POSTFIX ""
    EXPORT_NAME pyBART)
    set(BART_pyBART_TGT pyBART)
endif()

#==============================================
# TODO: Matlab code

if(BART_MATLAB)
  find_package(Matlab REQUIRED)
  if(MATLAB_FOUND)
    message(STATUS "MATLAB LIBRARIES FOUND: ${MATLAB_LIBRARIES_DIR}")
    include_directories(${MATLAB_INCLUDE_DIR})
    bart_add_executable(mat2cfl ${CMAKE_CURRENT_LIST_DIR}/src/mat2cfl.c ${LOG_BACKEND_SRCS})
    target_link_libraries(mat2cfl PRIVATE bartsupport ${MATLAB_MAT_LIBRARY} ${MATLAB_ENG_LIBRARY} ${MATLAB_MX_LIBRARY})
    set(BART_MATLAB_TGT mat2cfl)
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
    target_link_libraries(${test_name} PUBLIC bartsupport)
    target_compile_definitions(${test_name} PUBLIC UTESTS=${${test_name}_tests})
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()

# ------------------------------------------------------------------------------

##Testing broke in 2c8c1a27 when adding a test.

## Include integration testing
#if(NOT BART_MEMONLY_CFL)
#  add_subdirectory(tests)
#endif()

# ==============================================================================

include(${CMAKE_CURRENT_LIST_DIR}/cmake/bart_install.cmake)

# Packaging stuff (deb, rpm, windows installer)
include(${CMAKE_CURRENT_LIST_DIR}/packaging/CPackConfig.cmake)

# ==============================================================================

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

if(BART_GENERATE_DOC)
  add_custom_command(TARGET bart
    POST_BUILD
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/rules/update_commands.sh $<TARGET_FILE:bart> ${CMAKE_CURRENT_LIST_DIR}/doc/commands.txt ${ALLPROGS}
    )
endif()


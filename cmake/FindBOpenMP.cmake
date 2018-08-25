# Taken from https://github.com/Kitware/CMake for CMake 3.12
#
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#.rst:
# FindBOpenMP
# ----------
#
# Finds OpenMP support
#
# This module can be used to detect OpenMP support in a compiler.  If
# the compiler supports OpenMP, the flags required to compile with
# OpenMP support are returned in variables for the different languages.
# The variables may be empty if the compiler does not need a special
# flag to support OpenMP.
#
# Variables
# ^^^^^^^^^
#
# The module exposes the components ``C``, ``CXX``, and ``Fortran``.
# Each of these controls the various languages to search OpenMP support for.
#
# Depending on the enabled components the following variables will be set:
#
# ``OpenMP_FOUND``
#   Variable indicating that OpenMP flags for all requested languages have been found.
#   If no components are specified, this is true if OpenMP settings for all enabled languages
#   were detected.
# ``OpenMP_VERSION``
#   Minimal version of the OpenMP standard detected among the requested languages,
#   or all enabled languages if no components were specified.
#
# This module will set the following variables per language in your
# project, where ``<lang>`` is one of C, CXX, or Fortran:
#
# ``OpenMP_<lang>_FOUND``
#   Variable indicating if OpenMP support for ``<lang>`` was detected.
# ``OpenMP_<lang>_FLAGS``
#   OpenMP compiler flags for ``<lang>``, separated by spaces.
#
# For linking with OpenMP code written in ``<lang>``, the following
# variables are provided:
#
# ``OpenMP_<lang>_LIB_NAMES``
#   :ref:`;-list <CMake Language Lists>` of libraries for OpenMP programs for ``<lang>``.
# ``OpenMP_<libname>_LIBRARY``
#   Location of the individual libraries needed for OpenMP support in ``<lang>``.
# ``OpenMP_<lang>_LIBRARIES``
#   A list of libraries needed to link with OpenMP code written in ``<lang>``.
#
# Additionally, the module provides :prop_tgt:`IMPORTED` targets:
#
# ``OpenMP::OpenMP_<lang>``
#   Target for using OpenMP from ``<lang>``.
#
# Specifically for Fortran, the module sets the following variables:
#
# ``OpenMP_Fortran_HAVE_OMPLIB_HEADER``
#   Boolean indicating if OpenMP is accessible through ``omp_lib.h``.
# ``OpenMP_Fortran_HAVE_OMPLIB_MODULE``
#   Boolean indicating if OpenMP is accessible through the ``omp_lib`` Fortran module.
#
# The module will also try to provide the OpenMP version variables:
#
# ``OpenMP_<lang>_SPEC_DATE``
#   Date of the OpenMP specification implemented by the ``<lang>`` compiler.
# ``OpenMP_<lang>_VERSION_MAJOR``
#   Major version of OpenMP implemented by the ``<lang>`` compiler.
# ``OpenMP_<lang>_VERSION_MINOR``
#   Minor version of OpenMP implemented by the ``<lang>`` compiler.
# ``OpenMP_<lang>_VERSION``
#   OpenMP version implemented by the ``<lang>`` compiler.
#
# The specification date is formatted as given in the OpenMP standard:
# ``yyyymm`` where ``yyyy`` and ``mm`` represents the year and month of
# the OpenMP specification implemented by the ``<lang>`` compiler.

function(_BOPENMP_FLAG_CANDIDATES LANG)
  if(NOT BOpenMP_${LANG}_FLAG)
    unset(BOpenMP_FLAG_CANDIDATES)
    set(OMP_FLAG_GNU "-fopenmp")
    set(OMP_FLAG_Clang "-fopenmp=libomp" "-fopenmp=libiomp5" "-fopenmp")
    set(OMP_FLAG_AppleClang "-Xclang -fopenmp")
    set(OMP_FLAG_HP "+Oopenmp")
    if(WIN32)
      set(OMP_FLAG_Intel "-Qopenmp")
    elseif(CMAKE_${LANG}_COMPILER_ID STREQUAL "Intel" AND
           "${CMAKE_${LANG}_COMPILER_VERSION}" VERSION_LESS "15.0.0.20140528")
      set(OMP_FLAG_Intel "-openmp")
    else()
      set(OMP_FLAG_Intel "-qopenmp")
    endif()
    set(OMP_FLAG_MIPSpro "-mp")
    set(OMP_FLAG_MSVC "-openmp")
    set(OMP_FLAG_PathScale "-openmp")
    set(OMP_FLAG_NAG "-openmp")
    set(OMP_FLAG_Absoft "-openmp")
    set(OMP_FLAG_PGI "-mp")
    set(OMP_FLAG_Flang "-fopenmp")
    set(OMP_FLAG_SunPro "-xopenmp")
    set(OMP_FLAG_XL "-qsmp=omp")
    # Cray compiler activate OpenMP with -h omp, which is enabled by default.
    set(OMP_FLAG_Cray " " "-h omp")

    # If we know the correct flags, use those
    if(DEFINED OMP_FLAG_${CMAKE_${LANG}_COMPILER_ID})
      set(BOpenMP_FLAG_CANDIDATES "${OMP_FLAG_${CMAKE_${LANG}_COMPILER_ID}}")
    # Fall back to reasonable default tries otherwise
    else()
      set(BOpenMP_FLAG_CANDIDATES "-openmp" "-fopenmp" "-mp" " ")
    endif()
    set(BOpenMP_${LANG}_FLAG_CANDIDATES "${BOpenMP_FLAG_CANDIDATES}" PARENT_SCOPE)
  else()
    set(BOpenMP_${LANG}_FLAG_CANDIDATES "${BOpenMP_${LANG}_FLAG}" PARENT_SCOPE)
  endif()
endfunction()

set(BOpenMP_C_CXX_TEST_SOURCE
"
#include <omp.h>
int main() {
#ifdef _OPENMP
  omp_get_max_threads();
  return 0;
#else
  breaks_on_purpose
#endif
}
")

set(BOpenMP_Fortran_TEST_SOURCE
  "
      program test
      @BOpenMP_Fortran_INCLUDE_LINE@
  !$  integer :: n
      n = omp_get_num_threads()
      end program test
  "
)

function(_BOPENMP_WRITE_SOURCE_FILE LANG SRC_FILE_CONTENT_VAR SRC_FILE_NAME SRC_FILE_FULLPATH)
  set(WORK_DIR ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindBOpenMP)
  if("${LANG}" STREQUAL "C")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.c")
    file(WRITE "${SRC_FILE}" "${BOpenMP_C_CXX_${SRC_FILE_CONTENT_VAR}}")
  elseif("${LANG}" STREQUAL "CXX")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.cpp")
    file(WRITE "${SRC_FILE}" "${BOpenMP_C_CXX_${SRC_FILE_CONTENT_VAR}}")
  elseif("${LANG}" STREQUAL "Fortran")
    set(SRC_FILE "${WORK_DIR}/${SRC_FILE_NAME}.f90")
    file(WRITE "${SRC_FILE}_in" "${BOpenMP_Fortran_${SRC_FILE_CONTENT_VAR}}")
    configure_file("${SRC_FILE}_in" "${SRC_FILE}" @ONLY)
  endif()
  set(${SRC_FILE_FULLPATH} "${SRC_FILE}" PARENT_SCOPE)
endfunction()

include(CMakeParseImplicitLinkInfo)

function(_BOPENMP_GET_FLAGS LANG FLAG_MODE BOPENMP_FLAG_VAR BOPENMP_LIB_NAMES_VAR)
  _BOPENMP_FLAG_CANDIDATES("${LANG}")
  _BOPENMP_WRITE_SOURCE_FILE("${LANG}" "TEST_SOURCE" BOpenMPTryFlag _BOPENMP_TEST_SRC)

  unset(BOpenMP_VERBOSE_COMPILE_OPTIONS)
  if(WIN32)
    separate_arguments(BOpenMP_VERBOSE_OPTIONS WINDOWS_COMMAND "${CMAKE_${LANG}_VERBOSE_FLAG}")
  else()
    separate_arguments(BOpenMP_VERBOSE_OPTIONS UNIX_COMMAND "${CMAKE_${LANG}_VERBOSE_FLAG}")
  endif()
  foreach(_VERBOSE_OPTION IN LISTS BOpenMP_VERBOSE_OPTIONS)
    if(NOT _VERBOSE_OPTION MATCHES "^-Wl,")
      list(APPEND BOpenMP_VERBOSE_COMPILE_OPTIONS ${_VERBOSE_OPTION})
    endif()
  endforeach()

  foreach(BOPENMP_FLAG IN LISTS BOpenMP_${LANG}_FLAG_CANDIDATES)
    set(BOPENMP_FLAGS_TEST "${BOPENMP_FLAG}")
    if(BOpenMP_VERBOSE_COMPILE_OPTIONS)
      set(BOPENMP_FLAGS_TEST "${BOPENMP_FLAGS_TEST} ${BOpenMP_VERBOSE_COMPILE_OPTIONS}")
      # string(APPEND BOPENMP_FLAGS_TEST " ${BOpenMP_VERBOSE_COMPILE_OPTIONS}")
    endif()
    string(REGEX REPLACE "[-/=+]" "" BOPENMP_PLAIN_FLAG "${BOPENMP_FLAG}")
    try_compile( BOpenMP_COMPILE_RESULT_${FLAG_MODE}_${BOPENMP_PLAIN_FLAG} ${CMAKE_BINARY_DIR} ${_BOPENMP_TEST_SRC}
      CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${BOPENMP_FLAGS_TEST}"
      LINK_LIBRARIES ${CMAKE_${LANG}_VERBOSE_FLAG}
      OUTPUT_VARIABLE BOpenMP_TRY_COMPILE_OUTPUT
    )

    if(BOpenMP_COMPILE_RESULT_${FLAG_MODE}_${BOPENMP_PLAIN_FLAG})
      set("${BOPENMP_FLAG_VAR}" "${BOPENMP_FLAG}" PARENT_SCOPE)

      if(CMAKE_${LANG}_VERBOSE_FLAG)
        unset(BOpenMP_${LANG}_IMPLICIT_LIBRARIES)
        unset(BOpenMP_${LANG}_IMPLICIT_LINK_DIRS)
        unset(BOpenMP_${LANG}_IMPLICIT_FWK_DIRS)
        unset(BOpenMP_${LANG}_LOG_VAR)

        file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "Detecting ${LANG} OpenMP compiler ABI info compiled with the following output:\n${BOpenMP_TRY_COMPILE_OUTPUT}\n\n")

        cmake_parse_implicit_link_info("${BOpenMP_TRY_COMPILE_OUTPUT}"
          BOpenMP_${LANG}_IMPLICIT_LIBRARIES
          BOpenMP_${LANG}_IMPLICIT_LINK_DIRS
          BOpenMP_${LANG}_IMPLICIT_FWK_DIRS
          BOpenMP_${LANG}_LOG_VAR
          "${CMAKE_${LANG}_IMPLICIT_OBJECT_REGEX}"
        )

        file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
        "Parsed ${LANG} OpenMP implicit link information from above output:\n${BOpenMP_${LANG}_LOG_VAR}\n\n")

        unset(_BOPENMP_LIB_NAMES)
        foreach(_BOPENMP_IMPLICIT_LIB IN LISTS BOpenMP_${LANG}_IMPLICIT_LIBRARIES)
          get_filename_component(_BOPENMP_IMPLICIT_LIB_DIR "${_BOPENMP_IMPLICIT_LIB}" DIRECTORY)
          get_filename_component(_BOPENMP_IMPLICIT_LIB_NAME "${_BOPENMP_IMPLICIT_LIB}" NAME)
          get_filename_component(_BOPENMP_IMPLICIT_LIB_PLAIN "${_BOPENMP_IMPLICIT_LIB}" NAME_WE)
          string(REGEX REPLACE "([][+.*?()^$])" "\\\\\\1" _BOPENMP_IMPLICIT_LIB_PLAIN_ESC "${_BOPENMP_IMPLICIT_LIB_PLAIN}")
          string(REGEX REPLACE "([][+.*?()^$])" "\\\\\\1" _BOPENMP_IMPLICIT_LIB_PATH_ESC "${_BOPENMP_IMPLICIT_LIB}")

	  list (FIND CMAKE_${LANG}_IMPLICIT_LINK_LIBRARIES "${_BOPENMP_IMPLICIT_LIB}" _index)
          if(NOT ( _index GREATER -1
            OR "${CMAKE_${LANG}_STANDARD_LIBRARIES}" MATCHES "(^| )(-Wl,)?(-l)?(${_BOPENMP_IMPLICIT_LIB_PLAIN_ESC}|${_BOPENMP_IMPLICIT_LIB_PATH_ESC})( |$)"
            OR "${CMAKE_${LANG}_LINK_EXECUTABLE}" MATCHES "(^| )(-Wl,)?(-l)?(${_BOPENMP_IMPLICIT_LIB_PLAIN_ESC}|${_BOPENMP_IMPLICIT_LIB_PATH_ESC})( |$)" ) )
            if(_BOPENMP_IMPLICIT_LIB_DIR)
              set(BOpenMP_${_BOPENMP_IMPLICIT_LIB_PLAIN}_LIBRARY "${_BOPENMP_IMPLICIT_LIB}" CACHE FILEPATH
                "Path to the ${_BOPENMP_IMPLICIT_LIB_PLAIN} library for OpenMP")
            else()
              find_library(BOpenMP_${_BOPENMP_IMPLICIT_LIB_PLAIN}_LIBRARY
                NAMES "${_BOPENMP_IMPLICIT_LIB_NAME}"
                DOC "Path to the ${_BOPENMP_IMPLICIT_LIB_PLAIN} library for OpenMP"
                HINTS ${BOpenMP_${LANG}_IMPLICIT_LINK_DIRS}
                CMAKE_FIND_ROOT_PATH_BOTH
                NO_DEFAULT_PATH
              )
            endif()
            mark_as_advanced(BOpenMP_${_BOPENMP_IMPLICIT_LIB_PLAIN}_LIBRARY)
            list(APPEND _BOPENMP_LIB_NAMES ${_BOPENMP_IMPLICIT_LIB_PLAIN})
          endif()
        endforeach()
        set("${BOPENMP_LIB_NAMES_VAR}" "${_BOPENMP_LIB_NAMES}" PARENT_SCOPE)
      else()
        set("${BOPENMP_LIB_NAMES_VAR}" "" PARENT_SCOPE)
      endif()
      break()
    elseif(CMAKE_${LANG}_COMPILER_ID STREQUAL "AppleClang"
      AND CMAKE_${LANG}_COMPILER_VERSION VERSION_GREATER_EQUAL "7.0")
      find_library(BOpenMP_libomp_LIBRARY
        NAMES omp gomp iomp5
        HINTS ${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}
      )
      mark_as_advanced(BOpenMP_libomp_LIBRARY)
      if(BOpenMP_libomp_LIBRARY)
        try_compile( BOpenMP_COMPILE_RESULT_${FLAG_MODE}_${BOPENMP_PLAIN_FLAG} ${CMAKE_BINARY_DIR} ${_BOPENMP_TEST_SRC}
          CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${BOPENMP_FLAGS_TEST}"
          LINK_LIBRARIES ${CMAKE_${LANG}_VERBOSE_FLAG} ${BOpenMP_libomp_LIBRARY}
          OUTPUT_VARIABLE BOpenMP_TRY_COMPILE_OUTPUT
        )
        if(BOpenMP_COMPILE_RESULT_${FLAG_MODE}_${BOPENMP_PLAIN_FLAG})
          set("${BOPENMP_FLAG_VAR}" "${BOPENMP_FLAG}" PARENT_SCOPE)
          set("${BOPENMP_LIB_NAMES_VAR}" "libomp" PARENT_SCOPE)
          break()
        endif()
      endif()
    endif()
    set("${BOPENMP_LIB_NAMES_VAR}" "NOTFOUND" PARENT_SCOPE)
    set("${BOPENMP_FLAG_VAR}" "NOTFOUND" PARENT_SCOPE)
  endforeach()
  unset(BOpenMP_VERBOSE_COMPILE_OPTIONS)
endfunction()

set(BOpenMP_C_CXX_CHECK_VERSION_SOURCE
"
#include <stdio.h>
#include <omp.h>
const char ompver_str[] = { 'I', 'N', 'F', 'O', ':', 'O', 'p', 'e', 'n', 'M',
                            'P', '-', 'd', 'a', 't', 'e', '[',
                            ('0' + ((_OPENMP/100000)%10)),
                            ('0' + ((_OPENMP/10000)%10)),
                            ('0' + ((_OPENMP/1000)%10)),
                            ('0' + ((_OPENMP/100)%10)),
                            ('0' + ((_OPENMP/10)%10)),
                            ('0' + ((_OPENMP/1)%10)),
                            ']', '\\0' };
int main()
{
  puts(ompver_str);
  return 0;
}
")

set(BOpenMP_Fortran_CHECK_VERSION_SOURCE
"
      program omp_ver
      @BOpenMP_Fortran_INCLUDE_LINE@
      integer, parameter :: zero = ichar('0')
      integer, parameter :: ompv = openmp_version
      character, dimension(24), parameter :: ompver_str =&
      (/ 'I', 'N', 'F', 'O', ':', 'O', 'p', 'e', 'n', 'M', 'P', '-',&
         'd', 'a', 't', 'e', '[',&
         char(zero + mod(ompv/100000, 10)),&
         char(zero + mod(ompv/10000, 10)),&
         char(zero + mod(ompv/1000, 10)),&
         char(zero + mod(ompv/100, 10)),&
         char(zero + mod(ompv/10, 10)),&
         char(zero + mod(ompv/1, 10)), ']' /)
      print *, ompver_str
      end program omp_ver
")

function(_BOPENMP_GET_SPEC_DATE LANG SPEC_DATE)
  _BOPENMP_WRITE_SOURCE_FILE("${LANG}" "CHECK_VERSION_SOURCE" BOpenMPCheckVersion _BOPENMP_TEST_SRC)
  set(BIN_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindBOpenMP/ompver_${LANG}.bin")
  string(REGEX REPLACE "[-/=+]" "" BOPENMP_PLAIN_FLAG "${BOPENMP_FLAG}")
  try_compile(BOpenMP_SPECTEST_${LANG}_${BOPENMP_PLAIN_FLAG} "${CMAKE_BINARY_DIR}" "${_BOPENMP_TEST_SRC}"
              CMAKE_FLAGS "-DCOMPILE_DEFINITIONS:STRING=${BOpenMP_${LANG}_FLAGS}"
              COPY_FILE ${BIN_FILE})
  if(${BOpenMP_SPECTEST_${LANG}_${BOPENMP_PLAIN_FLAG}})
    file(STRINGS ${BIN_FILE} specstr LIMIT_COUNT 1 REGEX "INFO:OpenMP-date")
    set(regex_spec_date ".*INFO:OpenMP-date\\[0*([^]]*)\\].*")
    if("${specstr}" MATCHES "${regex_spec_date}")
      set(${SPEC_DATE} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

macro(_BOPENMP_SET_VERSION_BY_SPEC_DATE LANG)
  set(BOpenMP_SPEC_DATE_MAP
    "201611=5.0"
    "201511=4.5"
    "201307=4.0"
    "201107=3.1"
    "200805=3.0"
    "200505=2.5"
    "200203=2.0"
    "200011=2.0"
    "199911=1.1"
    "199810=1.0"
    "199710=1.0"
  )

  if(BOpenMP_${LANG}_SPEC_DATE)
    string(REGEX MATCHALL "${BOpenMP_${LANG}_SPEC_DATE}=([0-9]+)\\.([0-9]+)" _version_match "${BOpenMP_SPEC_DATE_MAP}")
  else()
    set(_version_match "")
  endif()
  if(NOT _version_match STREQUAL "")
    set(BOpenMP_${LANG}_VERSION_MAJOR ${CMAKE_MATCH_1})
    set(BOpenMP_${LANG}_VERSION_MINOR ${CMAKE_MATCH_2})
    set(BOpenMP_${LANG}_VERSION "${BOpenMP_${LANG}_VERSION_MAJOR}.${BOpenMP_${LANG}_VERSION_MINOR}")
  else()
    unset(BOpenMP_${LANG}_VERSION_MAJOR)
    unset(BOpenMP_${LANG}_VERSION_MINOR)
    unset(BOpenMP_${LANG}_VERSION)
  endif()
  unset(_version_match)
  unset(BOpenMP_SPEC_DATE_MAP)
endmacro()

foreach(LANG IN ITEMS C CXX)
  if(CMAKE_${LANG}_COMPILER_LOADED)
    if(NOT DEFINED BOpenMP_${LANG}_FLAGS OR "${BOpenMP_${LANG}_FLAGS}" STREQUAL "NOTFOUND"
      OR NOT DEFINED BOpenMP_${LANG}_LIB_NAMES OR "${BOpenMP_${LANG}_LIB_NAMES}" STREQUAL "NOTFOUND")
      _BOPENMP_GET_FLAGS("${LANG}" "${LANG}" BOpenMP_${LANG}_FLAGS_WORK BOpenMP_${LANG}_LIB_NAMES_WORK)
    endif()
    set(BOpenMP_${LANG}_FLAGS "${BOpenMP_${LANG}_FLAGS_WORK}"
      CACHE STRING "${LANG} compiler flags for OpenMP parallelization")
    set(BOpenMP_${LANG}_LIB_NAMES "${BOpenMP_${LANG}_LIB_NAMES_WORK}"
      CACHE STRING "${LANG} compiler libraries for OpenMP parallelization")
    mark_as_advanced(BOpenMP_${LANG}_FLAGS BOpenMP_${LANG}_LIB_NAMES)
  endif()
endforeach()

if(CMAKE_Fortran_COMPILER_LOADED)
  if(NOT DEFINED BOpenMP_Fortran_FLAGS OR "${BOpenMP_Fortran_FLAGS}" STREQUAL "NOTFOUND"
    OR NOT DEFINED BOpenMP_Fortran_LIB_NAMES OR "${BOpenMP_Fortran_LIB_NAMES}" STREQUAL "NOTFOUND"
    OR NOT DEFINED BOpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(BOpenMP_Fortran_INCLUDE_LINE "use omp_lib\n      implicit none")
    _BOPENMP_GET_FLAGS("Fortran" "FortranHeader" BOpenMP_Fortran_FLAGS_WORK BOpenMP_Fortran_LIB_NAMES_WORK)
    if(BOpenMP_Fortran_FLAGS_WORK)
      set(BOpenMP_Fortran_HAVE_OMPLIB_MODULE TRUE CACHE BOOL INTERNAL "")
    endif()
    set(BOpenMP_Fortran_FLAGS "${BOpenMP_Fortran_FLAGS_WORK}"
      CACHE STRING "Fortran compiler flags for OpenMP parallelization")
    set(BOpenMP_Fortran_LIB_NAMES "${BOpenMP_Fortran_LIB_NAMES_WORK}"
      CACHE STRING "Fortran compiler libraries for OpenMP parallelization")
    mark_as_advanced(BOpenMP_Fortran_FLAGS BOpenMP_Fortran_LIB_NAMES)
  endif()

  if(NOT DEFINED BOpenMP_Fortran_FLAGS OR "${BOpenMP_Fortran_FLAGS}" STREQUAL "NOTFOUND"
    OR NOT DEFINED BOpenMP_Fortran_LIB_NAMES OR "${BOpenMP_Fortran_LIB_NAMES}" STREQUAL "NOTFOUND"
    OR NOT DEFINED BOpenMP_Fortran_HAVE_OMPLIB_HEADER)
    set(BOpenMP_Fortran_INCLUDE_LINE "implicit none\n      include 'omp_lib.h'")
    _BOPENMP_GET_FLAGS("Fortran" "FortranModule" BOpenMP_Fortran_FLAGS_WORK BOpenMP_Fortran_LIB_NAMES_WORK)
    if(BOpenMP_Fortran_FLAGS_WORK)
      set(BOpenMP_Fortran_HAVE_OMPLIB_HEADER TRUE CACHE BOOL INTERNAL "")
    endif()
    set(BOpenMP_Fortran_FLAGS "${BOpenMP_Fortran_FLAGS_WORK}"
      CACHE STRING "Fortran compiler flags for OpenMP parallelization")

    set(BOpenMP_Fortran_LIB_NAMES "${BOpenMP_Fortran_LIB_NAMES}"
      CACHE STRING "Fortran compiler libraries for OpenMP parallelization")
  endif()
  if(BOpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(BOpenMP_Fortran_INCLUDE_LINE "use omp_lib\n      implicit none")
  else()
    set(BOpenMP_Fortran_INCLUDE_LINE "implicit none\n      include 'omp_lib.h'")
  endif()
endif()

if(NOT BOpenMP_FIND_COMPONENTS)
  set(BOpenMP_FINDLIST C CXX Fortran)
else()
  set(BOpenMP_FINDLIST ${BOpenMP_FIND_COMPONENTS})
endif()

unset(_BOpenMP_MIN_VERSION)

include(FindPackageHandleStandardArgs)

foreach(LANG IN LISTS BOpenMP_FINDLIST)
  if(CMAKE_${LANG}_COMPILER_LOADED)
    if (NOT BOpenMP_${LANG}_SPEC_DATE AND BOpenMP_${LANG}_FLAGS)
      _BOPENMP_GET_SPEC_DATE("${LANG}" BOpenMP_${LANG}_SPEC_DATE_INTERNAL)
      set(BOpenMP_${LANG}_SPEC_DATE "${BOpenMP_${LANG}_SPEC_DATE_INTERNAL}" CACHE
        INTERNAL "${LANG} compiler's OpenMP specification date")
      _BOPENMP_SET_VERSION_BY_SPEC_DATE("${LANG}")
    endif()

    set(BOpenMP_${LANG}_FIND_QUIETLY ${BOpenMP_FIND_QUIETLY})
    set(BOpenMP_${LANG}_FIND_REQUIRED ${BOpenMP_FIND_REQUIRED})
    set(BOpenMP_${LANG}_FIND_VERSION ${BOpenMP_FIND_VERSION})
    set(BOpenMP_${LANG}_FIND_VERSION_EXACT ${BOpenMP_FIND_VERSION_EXACT})

    set(_BOPENMP_${LANG}_REQUIRED_VARS BOpenMP_${LANG}_FLAGS)
    if("${BOpenMP_${LANG}_LIB_NAMES}" STREQUAL "NOTFOUND")
      set(_BOPENMP_${LANG}_REQUIRED_LIB_VARS BOpenMP_${LANG}_LIB_NAMES)
    else()
      foreach(_BOPENMP_IMPLICIT_LIB IN LISTS BOpenMP_${LANG}_LIB_NAMES)
        list(APPEND _BOPENMP_${LANG}_REQUIRED_LIB_VARS BOpenMP_${_BOPENMP_IMPLICIT_LIB}_LIBRARY)
      endforeach()
    endif()

    find_package_handle_standard_args(BOpenMP_${LANG} FOUND_VAR BOpenMP_${LANG}_FOUND
      REQUIRED_VARS BOpenMP_${LANG}_FLAGS ${_BOPENMP_${LANG}_REQUIRED_LIB_VARS}
      VERSION_VAR BOpenMP_${LANG}_VERSION
      )

    if(BOpenMP_${LANG}_FOUND)
      if(DEFINED BOpenMP_${LANG}_VERSION)
        if(NOT _BOpenMP_MIN_VERSION OR _BOpenMP_MIN_VERSION VERSION_GREATER BOpenMP_${LANG}_VERSION)
          set(_BOpenMP_MIN_VERSION BOpenMP_${LANG}_VERSION)
        endif()
      endif()
      set(BOpenMP_${LANG}_LIBRARIES "")
      foreach(_BOPENMP_IMPLICIT_LIB IN LISTS BOpenMP_${LANG}_LIB_NAMES)
        list(APPEND BOpenMP_${LANG}_LIBRARIES "${BOpenMP_${_BOPENMP_IMPLICIT_LIB}_LIBRARY}")
      endforeach()

      if(NOT TARGET OpenMP::OpenMP_${LANG})
	if(CMAKE_VERSION VERSION_LESS 3.0)
          add_library(OpenMP::OpenMP_${LANG} UNKNOWN IMPORTED)
	  list(GET BOpenMP_${LANG}_LIB_NAMES 0 _name)
	  set_target_properties(OpenMP::OpenMP_${LANG}
	    PROPERTIES
	    IMPORTED_LOCATION "${BOpenMP_${_name}_LIBRARY}")
	else()
          add_library(OpenMP::OpenMP_${LANG} INTERFACE IMPORTED)
	endif()
      endif()
      if(BOpenMP_${LANG}_FLAGS)
	if(WIN32)
          separate_arguments(_BOpenMP_${LANG}_OPTIONS WINDOWS_COMMAND "${BOpenMP_${LANG}_FLAGS}")
	else()
          separate_arguments(_BOpenMP_${LANG}_OPTIONS UNIX_COMMAND "${BOpenMP_${LANG}_FLAGS}")
	endif()
	if(CMAKE_VERSION VERSION_LESS 3.3)
          set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
            INTERFACE_COMPILE_OPTIONS "${_BOpenMP_${LANG}_OPTIONS}")
      else()
          set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
            INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:${LANG}>:${_BOpenMP_${LANG}_OPTIONS}>")
	endif()
        unset(_BOpenMP_${LANG}_OPTIONS)
      endif()
      if(BOpenMP_${LANG}_LIBRARIES)
        set_property(TARGET OpenMP::OpenMP_${LANG} PROPERTY
          INTERFACE_LINK_LIBRARIES "${BOpenMP_${LANG}_LIBRARIES}")
      endif()
    endif()
  endif()
endforeach()

unset(_BOpenMP_REQ_VARS)
foreach(LANG IN ITEMS C CXX Fortran)
  list (FIND BOpenMP_FIND_COMPONENTS "${LANG}" _index)

  if((NOT BOpenMP_FIND_COMPONENTS AND CMAKE_${LANG}_COMPILER_LOADED) OR _index GREATER -1)
    list(APPEND _BOpenMP_REQ_VARS "BOpenMP_${LANG}_FOUND")
  endif()
endforeach()

find_package_handle_standard_args(BOpenMP FOUND_VAR BOpenMP_FOUND
    REQUIRED_VARS ${_BOpenMP_REQ_VARS}
    VERSION_VAR ${_BOpenMP_MIN_VERSION}
    HANDLE_COMPONENTS)

set(BOPENMP_FOUND ${BOpenMP_FOUND})

if(CMAKE_Fortran_COMPILER_LOADED AND BOpenMP_Fortran_FOUND)
  if(NOT DEFINED BOpenMP_Fortran_HAVE_OMPLIB_MODULE)
    set(BOpenMP_Fortran_HAVE_OMPLIB_MODULE FALSE CACHE BOOL INTERNAL "")
  endif()
  if(NOT DEFINED BOpenMP_Fortran_HAVE_OMPLIB_HEADER)
    set(BOpenMP_Fortran_HAVE_OMPLIB_HEADER FALSE CACHE BOOL INTERNAL "")
  endif()
endif()

if(NOT ( CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED OR CMAKE_Fortran_COMPILER_LOADED ))
  message(SEND_ERROR "FindBOpenMP requires the C, CXX or Fortran languages to be enabled")
endif()

unset(BOpenMP_C_CXX_TEST_SOURCE)
unset(BOpenMP_Fortran_TEST_SOURCE)
unset(BOpenMP_C_CXX_CHECK_VERSION_SOURCE)
unset(BOpenMP_Fortran_CHECK_VERSION_SOURCE)
unset(BOpenMP_Fortran_INCLUDE_LINE)

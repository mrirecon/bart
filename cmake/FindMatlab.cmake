# - this module looks for Matlab
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:   required libraries: libmex, etc
#  MATLAB_MEX_LIBRARY: path to libmex.lib
#  MATLAB_MX_LIBRARY:  path to libmx.lib
#  MATLAB_MAT_LIBRARY:  path to libmat.lib # added
#  MATLAB_ENG_LIBRARY: path to libeng.lib
#  MATLAB_ROOT: path to Matlab's root directory

# This file is part of Gerardus
#
# This is a derivative work of file FindMatlab.cmake released with
# CMake v2.8, because the original seems to be a bit outdated and
# doesn't work with my Windows XP and Visual Studio 10 install
#
# (Note that the original file does work for Ubuntu Natty)
#
# Author: Ramon Casero <rcasero@gmail.com>, Tom Doel
# Version: 0.2.3
# $Rev$
# $Date$
#
# The original file was copied from an Ubuntu Linux install
# /usr/share/cmake-2.8/Modules/FindMatlab.cmake

#=============================================================================
# Copyright 2005-2009 Kitware, Inc.
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of CMake, substitute the full
#  License text for the above reference.)

set(MATLAB_FOUND 0)
if(WIN32)
  # Search for a version of Matlab available, starting from the most modern one to older versions
  foreach(MATVER "7.14" "7.11" "7.10" "7.9" "7.8" "7.7" "7.6" "7.5" "7.4")
    if((NOT DEFINED MATLAB_ROOT)
        OR ("${MATLAB_ROOT}" STREQUAL "")
        OR ("${MATLAB_ROOT}" STREQUAL "/registry"))
      get_filename_component(MATLAB_ROOT
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MathWorks\\MATLAB\\${MATVER};MATLABROOT]"
        ABSOLUTE)
      set(MATLAB_VERSION ${MATVER})
    endif()
  endforeach()

  # Directory name depending on whether the Windows architecture is 32
  # bit or 64 bit
  set(CMAKE_SIZEOF_VOID_P 8) # Note: For some wierd reason this variable is undefined in my system...
  if(CMAKE_SIZEOF_VOID_P MATCHES "4")
    set(WINDIR "win32")
  elseif(CMAKE_SIZEOF_VOID_P MATCHES "8")
    set(WINDIR "win64")
  else()
    message(FATAL_ERROR
      "CMAKE_SIZEOF_VOID_P (${CMAKE_SIZEOF_VOID_P}) doesn't indicate a valid platform")
  endif()

  # Folder where the MEX libraries are, depending of the Windows compiler
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
    set(MATLAB_LIBRARIES_DIR "${MATLAB_ROOT}/extern/lib/${WINDIR}/microsoft/msvc60")
  elseif(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
    # Assume people are generally using Visual Studio 7.1,
    # if using 7.0 need to link to: ../extern/lib/${WINDIR}/microsoft/msvc70
    set(MATLAB_LIBRARIES_DIR "${MATLAB_ROOT}/extern/lib/${WINDIR}/microsoft/msvc71")
    # set(MATLAB_LIBRARIES_DIR "${MATLAB_ROOT}/extern/lib/${WINDIR}/microsoft/msvc70")
  elseif(${CMAKE_GENERATOR} MATCHES "Borland")
    # Assume people are generally using Borland 5.4,
    # if using 7.0 need to link to: ../extern/lib/${WINDIR}/microsoft/msvc70
    set(MATLAB_LIBRARIES_DIR "${MATLAB_ROOT}/extern/lib/${WINDIR}/microsoft/bcc54")
    # set(MATLAB_LIBRARIES_DIR "${MATLAB_ROOT}/extern/lib/${WINDIR}/microsoft/bcc50")
    # set(MATLAB_LIBRARIES_DIR "${MATLAB_ROOT}/extern/lib/${WINDIR}/microsoft/bcc51")
  elseif(${CMAKE_GENERATOR} MATCHES "Visual Studio*")
    # If the compiler is Visual Studio, but not any of the specific
    # versions above, we try our luck with the microsoft directory
    set(MATLAB_LIBRARIES_DIR "${MATLAB_ROOT}/extern/lib/${WINDIR}/microsoft/")
  else()
    message(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
  endif()

else()

  if((NOT DEFINED MATLAB_ROOT) OR ("${MATLAB_ROOT}" STREQUAL ""))
    if(APPLE)
      # If this is a Mac and the attempts to find MATLAB_ROOT have so far failed,
      # we look in the applications folder
        # Search for a version of Matlab available, starting from the most modern one to older versions
        foreach(MATVER
                       "R2017b" "R2017a"
                       "R2016b" "R2016a"
                       "R2015b" "R2015a"
                       "R2014b" "R2014a"
                       "R2013b" "R2013a"
                       "R2012b" "R2012a"
                       "R2011b" "R2011a"
                       "R2010b" "R2010a"
                       "R2009b" "R2009a"
                       "R2008b")
          if(EXISTS /Applications/MATLAB_${MATVER}.app)
            set(MATLAB_ROOT /Applications/MATLAB_${MATVER}.app)
          endif()
        endforeach()
    endif()
    ## Search for matlab
    find_program(MATLAB_EXEC
               NAMES matlab
               HINTS ENV PATH
               PATHS ${MATLAB_ROOT}/bin
               DOC "The command line matlab program"
               )
    get_filename_component(MATLAB_EXEC "${MATLAB_EXEC}" REALPATH)
    get_filename_component(MATLAB_EXEC_DIR "${MATLAB_EXEC}" DIRECTORY)
    if(NOT MATLAB_ROOT)
      get_filename_component(MATLAB_ROOT "${MATLAB_EXEC_DIR}" DIRECTORY)
    endif()
    if(NOT MATLAB_EXEC)
      message(FATAL_ERROR "Matlab not found")
    endif()
  endif()

  # Check if this is a Mac
  if(APPLE)
    set(LIBRARY_EXTENSION dylib)
  else()
    set(LIBRARY_EXTENSION so)
  endif()

  find_program( MATLAB_MEX_PATH mex
             HINTS ENV PATH
             PATHS ${MATLAB_ROOT}/bin
             DOC "The mex program path"
            )

  find_program( MATLAB_MEXEXT_PATH mexext
             HINTS ENV PATH
             PATHS ${MATLAB_ROOT}/bin
             DOC "The mexext program path"
            )

  #Get default mex extentension
  execute_process(
	COMMAND ${MATLAB_MEXEXT_PATH}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE MATLAB_MEX_EXT
    )
  ## Remove the mex prefix to find the platform name
  if(APPLE)
    string(REPLACE "mex" "" MATLAB_PLATFORM_DIR ${MATLAB_MEX_EXT})
  else()
    string(REPLACE "mex" "glnx" MATLAB_PLATFORM_DIR ${MATLAB_MEX_EXT})
  endif()
  set(MATLAB_LIBRARIES_DIR ${MATLAB_ROOT}/bin/${MATLAB_PLATFORM_DIR})
endif()

# Get path to the MEX libraries
find_library(MATLAB_MEX_LIBRARY
             NAMES libmex.${LIBRARY_EXTENSION}
             PATHS ${MATLAB_LIBRARIES_DIR}
             NO_DEFAULT_PATH
             )

find_library(MATLAB_MX_LIBRARY
             NAMES libmx.${LIBRARY_EXTENSION}
             PATHS ${MATLAB_LIBRARIES_DIR}
             NO_DEFAULT_PATH
             )
find_library(MATLAB_MAT_LIBRARY
             NAMES libmat.${LIBRARY_EXTENSION}
             PATHS ${MATLAB_LIBRARIES_DIR}
             NO_DEFAULT_PATH
             )
find_library(MATLAB_ENG_LIBRARY
             NAMES libeng.${LIBRARY_EXTENSION}
             PATHS ${MATLAB_LIBRARIES_DIR}
             NO_DEFAULT_PATH
             )

# Get path to the include directory
find_path(MATLAB_INCLUDE_DIR
  NAMES "mex.h"
  PATHS "${MATLAB_ROOT}/extern/include"
  )


# This is common to UNIX and Win32:
set(MATLAB_LIBRARIES
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MAT_LIBRARY}
  ${MATLAB_ENG_LIBRARY}
)

if(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  set(MATLAB_FOUND 1)
endif()

mark_as_advanced(
  MATLAB_LIBRARIES
  MATLAB_MX_LIBRARY
  MATLAB_MEX_LIBRARY
  MATLAB_MAT_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_FOUND
  MATLAB_ROOT
  MATLAB_MEX_PATH
  MATLAB_MEXEXT_PATH
  MATLAB_MEX_EXT
)

#####################
#####################
# Provide a macro to build the mex files from
# within CMake
#####################
# BuildMex.cmake
# \author Kent Williams norman-k-williams@uiowa.edu
# \author Hans J. Johnson hans-johnson@uiowa.edu
include(CMakeParseArguments)
include_directories(${MATLAB_INCLUDE_DIR})
#
# BuildMex -- arguments
# MEXNAME = root of mex library name
# TARGETDIR = location for the mex library files to be created
# SOURCE = list of source files
# LIBRARIES = libraries needed to link mex library
macro(BuildMex)
  set(oneValueArgs MEXNAME TARGETDIR)
  set(multiValueArgs SOURCE LIBRARIES)
  cmake_parse_arguments(BuildMex "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # message("MEXNAME=${BuildMex_MEXNAME} SOURCE=${BuildMex_SOURCE} LIBRARIES=${BuildMex_LIBRARIES}")
  set_source_files_properties(${BuildMex_SOURCE} COMPILE_DEFINITIONS -DMATLAB_MEX_FILE )
  add_library(${BuildMex_MEXNAME} SHARED ${BuildMex_SOURCE})
  set_target_properties(${BuildMex_MEXNAME} PROPERTIES
    SUFFIX ".${MATLAB_MEX_EXT}"
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    LIBRARY_OUTPUT_DIRECTORY "${BuildMex_TARGETDIR}"
    )
  target_link_libraries(${BuildMex_MEXNAME} ${BuildMex_LIBRARIES} ${MATLAB_MEX_LIBRARY} ${MATLAB_MX_LIBRARY} ${MATLAB_ENG_LIBRARY})
endmacro(BuildMex)

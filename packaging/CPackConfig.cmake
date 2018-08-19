# Copyright 2018. Damien Nguyen <damien.nguyen@alumni.epfl.ch>
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
# \author Damien Nguyen <damien.nguyen@alumni.epfl.ch>

# ==============================================================================
#
# Inspired by https://github.com/kracejic/cleanCppProject
# Copyright (c) 2018 Kracejic
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

include(InstallRequiredSystemLibraries)

set(CPACK_PACKAGE_VENDOR "UC Berkeley")
set(CPACK_PACKAGE_CONTACT "Martin Uecker <martin.uecker@med.uni-goettingen.de>")
set(HOMEPAGE "https://github.com/mrirecon/bart")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "The Berkeley Advanced Reconstruction Toolbox (BART) toolbox is a free and open-source image-reconstruction framework for Computational Magnetic Resonance Imaging.")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${PROJECT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/packaging)
set(CPACK_PACKAGE_INSTALL_DIRECTORY "${CMAKE_PROJECT_NAME}")

# Generate .txt license file for CPack (PackageMaker requires a file extension)
configure_file(${PROJECT_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt @ONLY)

# set human names to exetuables
set(CPACK_PACKAGE_EXECUTABLES "bart;bart")
set(CPACK_CREATE_DESKTOP_LINKS "bart")
if(TARGET mat2cfl)
  list(APPEND CPACK_PACKAGE_EXECUTABLES mat2cfl mat2cfl)
  list(APPEND CPACK_CREATE_DESKTOP_LINKS mat2cfl)
endif()
set(CPACK_STRIP_FILES TRUE)

# ==============================================================================

if(WIN32 AND NOT UNIX)
    #---------------------------------------------------------------------------
    # Windows specific
    set(CPACK_GENERATOR "STGZ;ZIP")
    message(STATUS "Package generation - Windows")
    message(STATUS "   + STGZ                                 YES ")
    message(STATUS "   + ZIP                                  YES ")

    # NSIS windows installer
    find_program(NSIS_PATH nsis PATH_SUFFIXES nsis)
    if(NSIS_PATH)
        set(CPACK_GENERATOR "${CPACK_GENERATOR};NSIS")
        message(STATUS "   + NSIS                                 YES ")
        # Note: There is a bug in NSI that does not handle full unix paths
	#       properly.
	#       Make sure there is at least one set of four (4) backlasshes.
        set(CPACK_NSIS_DISPLAY_NAME ${CPACK_PACKAGE_NAME})
        # Icon of the installer
        set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}\\\\packaging\\\\bart.ico")
        set(CPACK_NSIS_HELP_LINK "http:\\\\\\\\www.mrirecon.github.io\\\\bart")
        set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
        set(CPACK_NSIS_MODIFY_PATH ON)
    else()
        message(STATUS "   + NSIS                                 NO ")
    endif()

    set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}\\\\bart.png")

    # Configure file with right path, place the result to PROJECT_BINARY_DIR.
    # When ${PROJECT_BINARY_DIR}/bart.icon.rc is added to an executable
    # it will have icon specified in bart.icon.in.rc
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/packaging/bart.icon.in.rc
        ${PROJECT_BINARY_DIR}/bart.icon.rc)

elseif(APPLE)
    #---------------------------------------------------------------------------
    # Apple specific
    set(CPACK_GENERATOR "ZIP;STGZ")
    message(STATUS "Package generation - MacOSX")
    message(STATUS "   + Application Bundle                   NO ")
    message(STATUS "   + ZIP                                  YES ")
else()
    #---------------------------------------------------------------------------
    # Linux specific
    set(CPACK_GENERATOR "DEB;TBZ2;STGZ")
    message(STATUS "Package generation - UNIX")
    message(STATUS "   + DEB                                  YES ")
    message(STATUS "   + TBZ2                                 YES ")
    message(STATUS "   + STGZ                                 YES ")

    find_program(RPMBUILD_PATH rpmbuild)
    if(RPMBUILD_PATH)
        message(STATUS "   + RPM                                  YES ")
        set(CPACK_GENERATOR "${CPACK_GENERATOR};RPM")
        set(CPACK_RPM_PACKAGE_LICENSE "3-Clause BSD License")
        set(CPACK_RPM_PACKAGE_REQUIRES "fftw3, blas")
	if(NOT BART_DISABLE_PNG)
	  set(CPACK_RPM_PACKAGE_REQUIRES
	    "${CPACK_RPM_PACKAGE_REQUIRES}, libpng")
	endif()
	# if(NOT BART_NO_LAPACKE)
	#   set(CPACK_RPM_PACKAGE_REQUIRES
	#     "${CPACK_RPM_PACKAGE_REQUIRES}, liblapacke")
	# endif()

        # exclude folders which clash with default ones
        set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST
            ${CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST}
            /usr
            /usr/bin
            /usr/share
            /usr/share/applications
            /usr/share/doc
            /usr/share/icons
            /usr/share/icons/hicolor
            /usr/share/icons/hicolor/256x256
            /usr/share/icons/hicolor/256x256/apps
            /usr/share/icons/gnome
            /usr/share/icons/gnome/256x256
            /usr/share/icons/gnome/256x256/apps)
    else()
        message(STATUS "   + RPM                                  NO ")
    endif()

    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION TRUE)
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${HOMEPAGE}")
    # set(CPACK_DEBIAN_COMPRESSION_TYPE "xz")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libfftw3-dev, libblas3") # FIXME: should really be something like libfftw3-3
    if(NOT BART_DISABLE_PNG)
      set(CPACK_DEBIAN_PACKAGE_DEPENDS
	"${CPACK_DEBIAN_PACKAGE_DEPENDS}, libpng-dev") # FIXME: should really be something like libpng16
    endif()
    # if(NOT BART_NO_LAPACKE)
    #   set(CPACK_DEBIAN_PACKAGE_DEPENDS
    # 	"${CPACK_DEBIAN_PACKAGE_DEPENDS}, liblapacke")
    # endif()

    # Icon and app shortcut for Linux systems
    # Note: .desktop file must have same name as executable
    # install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/bart.desktop
    #     DESTINATION share/applications/
    #     PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    #     )
    # install(FILES ${PROJECT_SOURCE_DIR}/packaging/bart.png
    #     DESTINATION share/icons/hicolor/256x256/apps/
    #     PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    #     )
    # install(FILES ${PROJECT_SOURCE_DIR}/packaging/bart.png
    #     DESTINATION share/icons/gnome/256x256/apps/
    #     PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    #     )
    # License file
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        RENAME copyright)
    # set package icon
    set(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/bart.png")
endif()

# ==============================================================================

include(CPack)



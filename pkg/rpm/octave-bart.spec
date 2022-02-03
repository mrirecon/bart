%global octpkg bart
Name:           octave-%{octpkg}
Version:        {{{ bart_version }}}
Release:        {{{ bart_release }}}%{?dist}
Epoch:          1
Summary:        Octave bindings for BART

License:        BSD
URL:            https://mrirecon.github.io/bart
VCS:            {{{ git_dir_vcs }}}
Source0:        {{{ git_archive path=. source_name=octave-bart dir_name=octave-bart }}}
BuildArch:      noarch

BuildRequires:  octave-devel

Requires:       bart, octave
Requires(post): octave
Requires(postun): octave

%description
The Berkeley Advanced Reconstruction Toolbox (BART) is a free and open-source image-reconstruction framework for Computational Magnetic Resonance Imaging.

This package provides Octave bindings for BART. 

%prep
{{{ git_setup_macro dir_name=octave-bart }}}
# files that belong inside an octave pkg according to https://octave.org/doc/v4.4.0/Creating-Packages.html
mkdir matlab/inst
mv matlab/*.m matlab/inst
cp LICENSE matlab/COPYING
cat > matlab/DESCRIPTION  <<EOF
Name: %{octpkg}
Version: %{version}
Date:  %{build_timestamp}
Author: See https://mrirecon.github.io/bart/
Maintainer: Philip Schaten
Title: %{summary} 
Description: %{summary}
License: %{license}
Categories: MRI
EOF


%build
mkdir -p %{_builddir}/%{buildsubdir}/build/
tar cvf %{_builddir}/%{buildsubdir}/build/%{octpkg}-%{version}-*.tar.gz matlab

%install
%octave_pkg_install


%post
%octave_cmd pkg rebuild

%preun
%octave_pkg_preun

%postun
%octave_cmd pkg rebuild


%files
%dir %{octpkgdir}
%{octpkgdir}/*.m
%doc %{octpkgdir}/doc-cache
%{octpkgdir}/packinfo


%changelog
{{{ git_dir_changelog }}}

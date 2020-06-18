Name:           bart
Version:        {{{ bart_git_version_dots }}} 
%define build_timestamp %{lua: print(os.date("%Y%m%d"))}
Release:        %{build_timestamp}%{?dist}
Summary:        tools for computational magnetic resonance imaging 

License:        BSD
URL:            https://mrirecon.github.io/bart
VCS:            {{{ git_dir_vcs }}}
Source0:        {{{ git_dir_pack source_name=bart dir_name=bart}}} 

BuildRequires:  gcc, make, fftw-devel, lapack-devel, openblas-devel, atlas-devel, libpng-devel
Requires:       fftw, lapack, openblas, atlas, libpng

%description
The Berkeley Advanced Reconstruction Toolbox (BART) is a free and open-source image-reconstruction framework for Computational Magnetic Resonance Imaging. It consists of a programming library and a toolbox of command-line programs. The library provides common operations on multi-dimensional arrays, Fourier and wavelet transforms, as well as generic implementations of iterative optimization algorithms. The command-line tools provide direct access to basic operations on multi-dimensional arrays as well as efficient implementations of many calibration and reconstruction algorithms for parallel imaging and compressed sensing.

# I suppose the binary shouldn't contain debug symbols by default
%global debug_package %{nil}


%prep
{{{ git_dir_setup_macro dir_name=bart }}}
# transfer .git-version information from rpkg-macro-expansion time to build time
echo {{{ bart_git_version }}} > version.txt

%build
make PARALLEL=1

%install
rm -rf $RPM_BUILD_ROOT
export
make PREFIX=usr DESTDIR=%{buildroot} install
mkdir -p %{buildroot}/usr/share/bash-completion/completions/
install scripts/bart_completion.sh %{buildroot}/usr/share/bash-completion/completions/
install -D doc/bart.1 %{buildroot}/%{_mandir}/man1/bart.1

%files
/usr/bin/bart
%license LICENSE
%{_mandir}/man1/bart.1*
%doc
/usr/share/doc/bart/README
/usr/share/doc/bart/applications.txt
/usr/share/doc/bart/bitmasks.txt
/usr/share/doc/bart/building.txt
/usr/share/doc/bart/commands.txt
/usr/share/doc/bart/debugging.txt
/usr/share/doc/bart/dimensions-and-strides.txt
/usr/share/doc/bart/fft.txt
/usr/share/doc/bart/references.txt
/usr/share/doc/bart/style.txt
/usr/share/bash-completion/completions/bart_completion.sh

%changelog
{{{ git_dir_changelog }}}

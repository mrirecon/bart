Name:       {{{ git_dir_name }}}
Version:        {{{ git_dir_version}}}
%define build_timestamp %{lua: print(os.date("%Y%m%d"))}
Release: %{build_timestamp}%{?dist}
Summary:        Toolbox for Computational Magnetic Resonance Imaging

License:        BSD
URL:            https://mrirecon.github.io/bart
VCS:            {{{ git_dir_vcs }}}
Source0:        {{{ git_dir_pack }}} 

BuildRequires:  gcc, make, fftw-devel, lapack-devel, openblas-devel, atlas-devel, libpng-devel
Requires:       fftw, lapack, openblas, atlas, libpng

%description
The library provides common operations on multi-dimensional arrays, Fourier and wavelet transforms, as well as generic implementations of iterative optimization algorithms. The command-line tools provide direct access to basic operations on multi-dimensional arrays as well as efficient implementations of many calibration and reconstruction algorithms for parallel imaging and compressed sensing. 

# I suppose the binary shouldn't contain debug symbols by default
%global debug_package %{nil}


%prep
{{{ git_dir_setup_macro }}}

%build
make PARALLEL=1

%install
rm -rf $RPM_BUILD_ROOT
export
make PREFIX=usr DESTDIR=%{buildroot} install

%files
/usr/bin/bart
%license LICENSE
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

%changelog
{{{ git_dir_changelog }}}

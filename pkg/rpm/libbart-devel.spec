Name:           libbart-devel
Version:        {{{ bart_version }}}
Release:        {{{ bart_release }}}%{?dist}
Epoch:          1
Summary:        Development files for BART 

License:        BSD
URL:            https://mrirecon.github.io/bart
VCS:            {{{ git_dir_vcs }}}
Source0:        {{{ git_archive path=. source_name=libbart-devel dir_name=libbart-devel }}}

BuildRequires:  gcc, make, fftw-devel, lapack-devel, openblas-devel, atlas-devel, libpng-devel

%description
The Berkeley Advanced Reconstruction Toolbox (BART) is a free and open-source image-reconstruction framework for Computational Magnetic Resonance Imaging.

This package provides headers and static libraries. 

# I suppose the binary shouldn't contain debug symbols by default
%global debug_package %{nil}

%prep
{{{ git_setup_macro dir_name=libbart-devel }}}

%build
export LDFLAGS="$LDFLAGS -Wl,--no-as-needed"
make PARALLEL=1

%install
rm -rf $RPM_BUILD_ROOT
while read line; do
src=$(cut -d' ' -f1 <<<"$line")
dst=%{buildroot}/$(cut -d' ' -f2 <<<"$line")
install -d "$dst"
install "$src" "$dst"
done < pkg/rpm/libbart-dev.install
# ^ Contents of https://salsa.debian.org/med-team/bart/-/blob/master/debian/libbart-dev.install

%files
/usr/include/bart/
/usr/lib/bart/
%license LICENSE

%changelog
{{{ git_dir_changelog }}}

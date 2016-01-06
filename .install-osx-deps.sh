brew update
brew tap Homebrew/homebrew-science
brew install gcc openmpi hdf5 netcdf python swig
pip install numpy
export PATH=/usr/local/bin:$PATH
export NETCDF_DIR=/usr/local
export BOOST_DIR=/usr/local

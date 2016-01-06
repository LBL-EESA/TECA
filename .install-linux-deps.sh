# Install necessary software.
sudo apt-get update -qq
sudo apt-get install -y cmake gcc libopenmpi-dev openmpi-bin libhdf5-openmpi-dev libnetcdf-dev libboost-program-options-dev swig liblapack-dev gfortran python-numpy

# Make symbolic links to third party software to match TECA's expectations.
export TECA_DIR=$PWD
mkdir -p $TECA_DIR/3rdparty/include
mkdir -p $TECA_DIR/3rdparty/lib
find /usr/include -type f -name "netcdf*" -exec ln -s {} $TECA_DIR/3rdparty/include \;
ln -s /usr/include/boost $TECA_DIR/3rdparty/include/boost
find /usr/lib/x86_64-linux-gnu -type f -name "libboost*" -exec ln -s {} $TECA_DIR/3rdparty/lib \;

# Tell TECA where to find NetCDF and Boost.
export NETCDF_DIR=$TECA_DIR/3rdparty
export BOOST_DIR=$TECA_DIR/3rdparty

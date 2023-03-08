# Copyright (c) 2020 Paul Ullrich 
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# NERSC Babbage Testbed

# C++ compiler without and with MPI
CXX=               CC
MPICXX=            CC

# NetCDF C library arguments
NETCDF_ROOT=       /opt/cray/pe/netcdf/4.9.0.1/GNU/9.1
NETCDF_CXXFLAGS=   -I$(NETCDF_ROOT)/include
NETCDF_LIBRARIES=  -lnetcdf -lnetcdf_c++
NETCDF_LDFLAGS=    -L$(NETCDF_ROOT)/lib

# DO NOT DELETE

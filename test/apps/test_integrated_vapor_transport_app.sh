#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_integrated_vapor_transport_app.sh [app prefix] "   \
         "[data root] [mpiexec] [num ranks]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

if [[ $# -eq 4 ]]
then
    mpi_exec=${3}
    test_cores=${4}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_integrated_vapor_transport --readers                                                                       \
    "r_0,${data_root}/HighResMIP/ECMWF-IFS-HR-SST-present/ua/ua_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc,ua"    \
    "r_1,${data_root}/HighResMIP/ECMWF-IFS-HR-SST-present/va/va_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc,va"    \
    "r_2,${data_root}/HighResMIP/ECMWF-IFS-HR-SST-present/hus/hus_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_1950-.*\\.nc,hus" \
    --steps_per_file 127 --verbose --output_file test_ivt_computation_app_output.nc

# run the diff
${app_prefix}/teca_cartesian_mesh_diff                        \
    "${data_root}/test_ivt_computation_app_ref.bin"           \
    "test_ivt_computation_app_output.*\.nc"

# clean up
rm test_ivt_computation_app_output*.nc

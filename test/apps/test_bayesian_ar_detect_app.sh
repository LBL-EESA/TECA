#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_bayesian_ar_detect_app.sh [app prefix] " \
         "[data root] [mpi exec] [test cores]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

launcher=
if [[ $# -eq 4 ]]
then
    mpi_exec=${3}
    test_cores=${4}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_bayesian_ar_detect                       \
    --verbose                                                           \
    --input_regex "${data_root}/ARTMIP_MERRA_2D_201702.*\.nc$"          \
    --cf_reader::filename_time_template "ARTMIP_MERRA_2D_%Y%m%d_%H.nc"  \
    --cf_reader::t_axis_variable ""                                     \
    --cf_reader::t_units "days since 1979-01-01 00:00:00"               \
    --output_file "${data_root}/test_bayesian_ar_detect_app_output.nc"

# run the diff
${app_prefix}/teca_cartesian_mesh_diff                      \
    0 "${data_root}/test_bayesian_ar_detect_app_output.nc"  \
    "${data_root}/test_bayesian_ar_detect_app_ref.bin"

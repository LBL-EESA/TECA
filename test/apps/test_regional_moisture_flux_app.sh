#!/bin/bash

if [[ $# < 3 ]]
then
    echo "usage: test_regional_moisture_flux_app.sh [app prefix] "   \
         "[data root] [num threads] [mpiexec] [num ranks]"
    exit -1
fi

app_prefix=${1}
data_root=${2}
n_threads=${3}

if [[ $# -eq 5 ]]
then
    mpi_exec=${4}
    test_cores=${5}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_regional_moisture_flux           \
    --input_regex "${data_root}/conus_masks/conus_ivt.*\.nc$"   \
    --shape_file ${data_root}/conus_masks/s_08mr23/s_08mr23     \
    --output_file test_conus_moisture_flux.csv                  \
    --ivt_u ivt_u --ivt_v ivt_v --bounds 230 298 22 52 0 0      \
    --verbose

do_test=1
if [[ $do_test -eq 0 ]]
then
    echo "regenerating baseline..."
    cp -vd ./test_conus_moisture_flux.csv \
         ${data_root}/test_conus_moisture_flux.csv
else
    # run the diff
    ${app_prefix}/teca_table_diff                       \
        "${data_root}/test_conus_moisture_flux.csv"     \
        "test_conus_moisture_flux.csv"

    # clean up
    rm test_conus_moisture_flux.csv
fi

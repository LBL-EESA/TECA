#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_deeplabv3p_ar_detect_app.sh [app prefix] " \
         "[data root] [mpi exec] [test cores]"
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
${launcher} ${app_prefix}/teca_deeplabv3p_ar_detect              \
    --input_regex "${data_root}/ARTMIP_MERRA_2D_2017-05.*\.nc$"  \
    --pytorch_deeplab_model ${data_root}/cascade_deeplab_IVT.pt  \
    --output_file test_deeplabv3p_ar_detect_app_output_%t%.nc

# run the diff
${app_prefix}/teca_cartesian_mesh_diff                           \
    "test_deeplabv3p_ar_detect_app_output.*\.nc"                 \
    "${data_root}/test_deeplabv3p_ar_detect_app_ref.nc"

# clean up
rm test_deeplabv3p_ar_detect_app_output*.nc

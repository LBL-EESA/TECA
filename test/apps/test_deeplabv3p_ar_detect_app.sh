#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_deeplabv3p_ar_detect_app.sh [app prefix] " \
         "[data root] [n threads] [mpi exec] [n ranks]"
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
${launcher} ${app_prefix}/teca_deeplabv3p_ar_detect              \
    --input_regex "${data_root}/ARTMIP_MERRA_2D_2017-05.*\.nc$"  \
    --pytorch_deeplab_model ${data_root}/cascade_deeplab_IVT.pt  \
    --n_threads ${n_threads}                                     \
    --output_file test_deeplabv3p_ar_detect_app_output_%t%.nc

# run the diff
${app_prefix}/teca_cartesian_mesh_diff                           \
    "${data_root}/test_deeplabv3p_ar_detect_app_ref.nc"          \
    "test_deeplabv3p_ar_detect_app_output.*\.nc"

# clean up
rm test_deeplabv3p_ar_detect_app_output*.nc

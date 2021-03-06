#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_deeplab_ar_detect_app.sh [app prefix] " \
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
${launcher} ${app_prefix}/teca_deeplab_ar_detect                 \
    --input_regex "${data_root}/ARTMIP_MERRA_2D_2017-05.*\.nc$"  \
    --pytorch_model ${data_root}/cascade_deeplab_IVT.pt          \
    --output_file test_deeplab_ar_detect_app_output_%t%.nc       \
    --steps_per_file 365 --n_threads ${n_threads}

# don't profile the diff
unset PROFILER_ENABLE

do_test=1
if [[ ${do_test} -eq 0 ]]
then
    # regenerate the baseline
    cp test_deeplab_ar_detect_app_output*.nc \
        ${data_root}/test_deeplab_ar_detect_app_ref.nc
else
    # run the diff
    ${app_prefix}/teca_cartesian_mesh_diff                                   \
        --reference_dataset "${data_root}/test_deeplab_ar_detect_app_ref.nc" \
        --test_dataset "test_deeplab_ar_detect_app_output.*\.nc"             \
        --arrays ar_probability ar_binary_tag --relative_tolerance 1e-4      \
        --absolute_tolerance 1e-6 --verbose

    # clean up
    rm test_deeplab_ar_detect_app_output*.nc
fi

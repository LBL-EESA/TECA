#!/bin/bash

if [[ $# < 3 ]]
then
    echo "usage: test_bayesian_ar_detect_app.sh [app prefix] "   \
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
${launcher} ${app_prefix}/teca_bayesian_ar_detect                \
    --input_regex "${data_root}/ARTMIP_MERRA_2D_2017-05.*\.nc$"  \
    --ar_weighted_variables IVT --segment_ar_probability         \
    --output_file test_bayesian_ar_detect_app_output_%t%.nc      \
    --steps_per_file 365 --n_threads ${n_threads} --verbose

do_test=1
if [[ $do_test -eq 0 ]]
then
    echo "regenerating baseline..."
    for f in `ls test_bayesian_ar_detect_app_output_*.nc`
    do
        ff=`echo $f | sed s/output/ref/g`
        cp -vd $f ${data_root}/$ff
    done
else
    # run the diff
    ${app_prefix}/teca_cartesian_mesh_diff                                          \
        --reference_dataset "${data_root}/test_bayesian_ar_detect_app_ref.*\.nc"    \
        --test_dataset "test_bayesian_ar_detect_app_output.*\.nc"                   \
        --arrays ar_probability ar_binary_tag ar_wgtd_IVT --verbose

    # clean up
    rm test_bayesian_ar_detect_app_output*.nc
fi

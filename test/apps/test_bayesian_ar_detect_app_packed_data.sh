#!/bin/bash

if [[ $# < 3 ]]
then
    echo "usage: test_bayesian_ar_detect_app_packed_data.sh " \
        "[app prefix] [data root] [n bard threads] [n writer threads] " \
        "[mpiexec] [num ranks]"
    exit -1
fi

app_prefix=${1}
data_root=${2}
n_bard_threads=${3}
n_writer_threads=${4}

if [[ $# -eq 6 ]]
then
    mpi_exec=${5}
    test_cores=${6}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_bayesian_ar_detect                                  \
    --input_regex "${data_root}/ERAinterim_1979-01-0.*\.nc$"                       \
    --x_axis_variable longitude --y_axis_variable latitude --z_axis_variable level \
    --wind_u uwnd --wind_v vwnd --specific_humidity shum --segment_ar_probability  \
    --compute_ivt --write_ivt --write_ivt_magnitude --steps_per_file 256           \
    --n_bard_threads ${n_bard_threads} --n_writer_threads ${n_writer_threads}      \
    --output_file test_bayesian_ar_detect_app_packed_data_output_%t%.nc            \
    --verbose 1

do_test=1
if [[ $do_test -eq 0 ]]
then
    echo "regenerating baseline..."
    for f in `ls test_bayesian_ar_detect_app_packed_data_output_*.nc`
    do
        ff=`echo $f | sed s/output/ref/g`
        cp -vd $f ${data_root}/$ff
    done
else
    # run the diff
    ${app_prefix}/teca_cartesian_mesh_diff                                                      \
        --reference_dataset "${data_root}/test_bayesian_ar_detect_app_packed_data_ref.*\.nc"    \
        --test_dataset "test_bayesian_ar_detect_app_packed_data_output.*\.nc"                   \
        --test_reader::x_axis_variable longitude --test_reader::y_axis_variable latitude        \
        --ref_reader::x_axis_variable longitude --ref_reader::y_axis_variable latitude          \
        --arrays ar_probability ar_binary_tag --verbose

    # clean up
    rm test_bayesian_ar_detect_app_packed_data_output*.nc
fi

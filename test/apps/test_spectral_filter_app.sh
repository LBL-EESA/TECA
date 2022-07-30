#!/bin/bash

if [[ $# -lt 7 ]]
then
    echo "usage: test_spectral_filter_app.sh [app prefix] " \
         "[data root] [input regex] [array name] [filter type] " \
         "[critical period] [period units] " \
         "[mpi exec] [test cores]"
    exit -1
fi

app_prefix=${1}
data_root=${2}
input_regex=${3}
array_name=${4}
filter_type=${5}
filter_order=${6}
critical_period=${7}
period_units=${8}


if [[ $# -eq 10 ]]
then
    mpi_exec=${9}
    test_cores=${10}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

test_name=test_spectral_filter
output_base=${test_name}_${filter_type}_${array_name}

# run the app
time ${launcher} ${app_prefix}/teca_spectral_filter                               \
    --input_regex "${data_root}/${input_regex}" --point_arrays ${array_name}      \
    --filter_type ${filter_type} --filter_order ${filter_order}                   \
    --critical_period ${critical_period} --critical_period_units ${period_units}  \
    --output_file "${output_base}_%t%.nc" --verbose 1

# don't profile the diff
unset PROFILER_ENABLE

do_test=1
if [[ $do_test -eq 0 ]]
then
    # update the baselines
    cp -vd ${output_base}_*.nc ${data_root}/
else
    # run the diff
    time ${app_prefix}/teca_cartesian_mesh_diff                     \
        --reference_dataset "${data_root}/${output_base}_.*\.nc"    \
        --test_dataset "${output_base}_.*\.nc"                      \
        --arrays ${array_name} --verbose                            \
        --relative_tolerance 1.e-5

    # clean up
    rm ${output_base}_*.nc
fi

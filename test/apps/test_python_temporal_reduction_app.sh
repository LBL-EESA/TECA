#!/bin/bash

if [[ $# -lt 8 ]]
then
    echo "usage: test_temporal_reduction_app.sh [app prefix] " \
         "[data root] [input regex] [array name] [interval] " \
         "[operator] [steps per file] [spatial partitioning] " \
         "[python impl] [mpi exec] [test cores] $#"
    exit -1
fi

app_prefix=${1}
data_root=${2}
input_regex=${3}
array_name=${4}
interval=${5}
operator=${6}
steps_per_file=${7}
spatial_part=${8}
if [[ ${spatial_part} -eq 1 ]]
then
    if [[ $# -eq 11 ]]
    then
        spatial_partitioning=--spatial_partitioning
    else
        spatial_partitioning="--spatial_partitioning --spatial_partitions 7"
    fi
fi
python_impl=${9}
if [[ ${python_impl} -eq 1 ]]
then
    version=--python_version
fi

if [[ $# -eq 11 ]]
then
    mpi_exec=${10}
    test_cores=${11}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

test_name=test_temporal_reduction
output_base=${test_name}_${array_name}_${interval}_${operator}

# run the app
time ${launcher} ${app_prefix}/teca_temporal_reduction           \
    --input_regex "${data_root}/${input_regex}" --interval ${interval}  \
    --operator ${operator} --point_arrays ${array_name}                 \
    --file_layout yearly --steps_per_file ${steps_per_file}             \
    --output_file "${output_base}_%t%.nc"                               \
    ${spatial_partitioning} ${version} --verbose 1

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

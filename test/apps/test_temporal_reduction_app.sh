#!/bin/bash

if [[ $# -lt 9 ]]
then
    echo "usage: test_temporal_reduction_app.sh [app prefix] " \
         "[data root] [input regex] [array name] [interval] " \
         "[operator] [steps per file] [n threads] [spatial partitioning] " \
         "[mpi exec] [test cores] $#"
    exit -1
fi

app_prefix=${1}
data_root=${2}
input_regex=${3}
array_name=${4}
if [[ ${5,,} =~ "_steps" ]]
then
    interval=n_steps
    n=$(echo $5 | cut -d "_" -f 1)
    number_of_steps="--number_of_steps $n"
else
    interval=${5}
fi

operator=${6}
steps_per_file=${7}
n_threads=${8}
if [[ ${9} -eq 1 ]]
then
    if [[ $# -eq 11 ]]
    then
        # with MPI let the partitioner decide
        spatial_partitioning=--spatial_partitioning
    else
        # without MPI, specify the number of partitions
        spatial_partitioning="--spatial_partitioning --spatial_partitions 7"
    fi
fi

# use mpi for this run
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
time ${launcher} ${app_prefix}/teca_temporal_reduction                  \
    --input_regex "${data_root}/${input_regex}" --interval ${interval}  \
    --operator ${operator} --point_arrays ${array_name}                 \
    --file_layout yearly --steps_per_file ${steps_per_file}             \
    --output_file "${output_base}_%t%.nc" --n_threads ${n_threads}      \
    ${spatial_partitioning} --verbose 1 ${number_of_steps}

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

#!/bin/bash

if [[ $# < 7 ]]
then
    echo "usage: test_temporal_reduction_app.sh [app prefix] " \
         "[data root] [input regex] [array name] [interval] " \
         "[operator] [steps per file] [mpi exec] [test cores]"
    exit -1
fi

app_prefix=${1}
data_root=${2}
input_regex=${3}
array_name=${4}
interval=${5}
operator=${6}
steps_per_file=${7}

if [[ $# -eq 9 ]]
then
    mpi_exec=${8}
    test_cores=${9}
    launcher="${mpi_exec} -n ${test_cores}"
fi

set -x

# run the app
${launcher} ${app_prefix}/teca_temporal_reduction \
    --input_regex "${data_root}/${input_regex}" --interval ${interval} \
    --operator ${operator} --point_arrays ${array_name} \
    --steps_per_file ${steps_per_file} --n_threads 2 --verbose 1 \
    --output_file "${array_name}_${interval}_${operator}_%t%.nc"

# run the diff
${app_prefix}/teca_cartesian_mesh_diff \
    --reference_dataset "${data_root}/test_temporal_reduction_app_${array_name}_${interval}_${operator}_.*\.nc" \
    --test_dataset "${array_name}_${interval}_${operator}_.*\.nc" \
    --arrays ${array_name} --verbose

# clean up
rm -f ${array_name}_${interval}_${operator}_*.nc

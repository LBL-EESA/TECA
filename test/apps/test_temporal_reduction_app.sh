#!/bin/bash

if [[ $# < 5 ]]
then
    echo "usage: test_temporal_reduction_app.sh [app prefix] "   \
         "[data root] [interval] [operator] [steps per file] " \
         "[mpi exec] [test cores]"
    exit -1
fi

app_prefix=${1}
data_root=${2}
interval=${3}
operator=${4}
steps_per_file=${5}

if [[ $# -eq 7 ]]
then
    mpi_exec=${6}
    test_cores=${7}
    launcher="${mpi_exec} -n ${test_cores}"
fi

input_regex="prw_hus_day_MRI-CGCM3_historical_r1i1p1_19500101-19501231\.nc"
output_prefix=prw

set -x

# run the app
${launcher} ${app_prefix}/teca_temporal_reduction                \
    --input_regex "${data_root}/${input_regex}"  \
    --interval ${interval} --operator ${operator} --arrays prw \
    --steps_per_file ${steps_per_file} --n_threads 2 --verbose 1 \
    --output_file "${output_prefix}_${interval}_${operator}_%t%.nc"

# run the diff
${app_prefix}/teca_cartesian_mesh_diff                           \
    "${output_prefix}_${interval}_${operator}_.*\.nc"                   \
    "${data_root}/${output_prefix}_${interval}_${operator}_.*\.nc"

# clean up
rm ${output_prefix}_${interval}_${operator}_*

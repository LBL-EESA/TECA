#!/bin/bash

if [[ $# < 2 ]]
then
    echo "usage: test_tc_trajectory_scalars_app.sh [app prefix] "     \
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

output_prefix="test_tc_trajectory_scalars_app_output"

set -x

# run the app
${launcher} ${app_prefix}/teca_tc_trajectory_scalars                  \
        "${data_root}/tracks_1990s_3hr_mdd_4800_median_in_cat_wr.bin" \
        ${output_prefix} --first_track 0 --last_track -1

# check if number of outputs are correct
output_list=($(ls ${output_prefix}*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 6 ]]
then
    echo "error: unexpected number of tc_trajectory_scalars outputs"
    exit 1
fi

# check if outputs aren't empty
for i in ${output_list[@]}
do
    if [[ ! -s $i ]]
    then
        echo "error: file '$i' is empty"
        exit 1
    fi
done

# clean up
rm ${output_prefix}*

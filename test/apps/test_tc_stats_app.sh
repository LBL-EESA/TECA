#!/bin/bash

if [[ $# -ne 2 ]]
then
    echo "usage: test_tc_stats_app.sh [app prefix] [data root]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

output_prefix="test_tc_stats_app_output"

set -x

# run the app
${app_prefix}/teca_tc_stats                                     \
    "${data_root}/cam5_1_amip_run2_tracks_2005_09.bin"          \
    ${output_prefix}

# run the diff
${app_prefix}/teca_table_diff                                   \
    "${data_root}/test_tc_stats_app_output_class_table_ref.bin" \
    test_tc_stats_app_output_class_table.csv

# check if number of outputs are correct
output_list=($(ls ${output_prefix}*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 13 ]]
then
    echo "error: unexpected number of tc_stats outputs"
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

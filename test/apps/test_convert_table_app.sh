#!/bin/bash

if [[ $# -ne 2 ]]
then
    echo "usage: test_convert_table_app.sh [app prefix] [data root]"
    exit -1
fi

app_prefix=${1}
data_root=${2}

output_prefix="test_convert_table_output.csv"

set -x

# run the app
${app_prefix}/teca_convert_table                       \
    "${data_root}/cam5_1_amip_run2_tracks_2005_09.bin" \
    ${output_prefix}

# check if number of outputs are correct
output_list=($(ls ${output_prefix}*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 1 ]]
then
    echo "error: unexpected number of convert_table outputs"
    exit 1
fi

if [[ ! -s ${output_prefix} ]]
then
    echo "error: file '${output_prefix}' is empty"
    exit 1
fi

# clean up
rm ${output_prefix}

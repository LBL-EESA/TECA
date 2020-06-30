#!/bin/bash

if [[ $# -ne 4 ]]
then
    echo "usage: test_tc_stats_app.sh" \
         "[app_prefix] [tracks_file] [output_file]" \
         "[output_ref]"
    exit 1
fi

# First command
app_name="teca_tc_stats"
app_prefix=${1}
tracks_file=${2}
output_file=${3}

# Second command
output_ref=${4}

app_exec="${app_prefix}/${app_name}"
verifier_exec="${app_prefix}/teca_table_diff"

${app_exec} ${tracks_file} ${output_file}

${verifier_exec} ${output_file}_class_table.bin ${output_ref}

output_list=($(ls ${output_file}_*))
output_len=${#output_list[@]}

if [[ ${output_len} -ne 14 ]]
then
    echo "error: unexpected number of tc_stats outputs"
    exit 1
fi

for i in ${output_list[@]}
do
    if [[ ! -s $i ]]
    then
        echo "error: file '$i' is empty"
        exit 1
    fi
done

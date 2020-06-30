#!/bin/bash

if [[ $# -ne 14 ]]
then
    echo "usage: test_tc_detect_app.sh" \
         "[app_prefix] [input_regex_option] [input_regex] " \
         "[candidate_file_option] [candidate_file]" \
         "[track_file_option] [track_file]" \
         "[lowest_lat_option] [lowest_lat]" \
         "[highest_lat_option] [highest_lat]" \
         "[candidate_ref] [track_ref]" \
         "[column name (to sort by)]"
    exit 1
fi

# First command
app_name="teca_tc_detect"
app_prefix=${1}
input_regex_option=${2}
input_regex=${3}
candidate_file_option=${4}
candidate_file=${5}
track_file_option=${6}
track_file=${7}
lowest_lat_option=${8}
lowest_lat=${9}
highest_lat_option=${10}
highest_lat=${11}

# Second command
candidate_ref="${12}"
track_ref="${13}"
column_name=${14}

app_exec="${app_prefix}/${app_name}"
verifier_exec="${app_prefix}/teca_table_diff"

eval ${app_exec} ${input_regex_option} ${input_regex} \
    ${candidate_file_option} ${candidate_file} \
    ${track_file_option} ${track_file} \
    ${lowest_lat_option} ${lowest_lat} \
    ${highest_lat_option} ${highest_lat}

${verifier_exec} ${candidate_file} \
    ${candidate_ref} ${column_name}

${verifier_exec} ${track_file} \
    ${track_ref} ${column_name}

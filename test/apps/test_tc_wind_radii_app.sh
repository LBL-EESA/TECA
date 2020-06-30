#!/bin/bash

if [[ $# -ne 20 ]]
then
    echo "usage: test_tc_wind_radii_app.sh" \
         "[app_prefix] [track_file_option] [track_file] " \
         "[wind_files_option] [wind_files] " \
         "[output_file_option] [output_file]" \
         "[track_mask_option] [track_mask]" \
         "[number_of_bins_option] [number_of_bins]" \
         "[profile_type_option] [profile_type]" \
         "[n_threads_option] [n_threads]" \
         "[first_track_option] [first_track]" \
         "[last_track_option] [last_track]" \
         "[output_ref]"
    exit 1
fi

# First command
app_name="teca_tc_wind_radii"
app_prefix=${1}
track_file_option=${2}
track_file=${3}
wind_files_option=${4}
wind_files="${5}"
output_file_option=${6}
output_file="${7}"
track_mask_option=${8}
track_mask="${9}"
number_of_bins_option=${10}
number_of_bins="${11}"
profile_type_option=${12}
profile_type="${13}"
n_threads_option=${14}
n_threads="${15}"
first_track_option=${16}
first_track="${17}"
last_track_option=${18}
last_track="${19}"

# Second command
output_ref="${20}"

app_exec="${app_prefix}/${app_name}"
verifier_exec="${app_prefix}/teca_table_diff"

${app_exec} ${track_file_option} ${track_file} \
    ${wind_files_option} ${wind_files} \
    ${output_file_option} ${output_file} \
    ${track_mask_option} ${track_mask} \
    ${number_of_bins_option} ${number_of_bins} \
    ${profile_type_option} ${profile_type} \
    ${n_threads_option} ${n_threads} \
    ${first_track_option} ${first_track} \
    ${last_track_option} ${last_track}

${verifier_exec} ${output_file} ${output_ref}

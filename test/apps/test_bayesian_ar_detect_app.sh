#!/bin/bash

if [[ $# -ne 13 ]]
then
    echo "usage: test_bayesian_ar_detect_app.sh" \
         "[app_prefix] [input_regex_option] [input_regex] " \
         "[f_time_template_option] [f_time_template]" \
         "[t_axis_var_option] [t_axis_var]" \
         "[t_units_option] [t_axis_var]" \
         "[t_units_option] [t_units]" \
         "[output_file_option] [output_file]" \
         "[output_file_type] [output_ref]"
    exit 1
fi

# First command
app_name="teca_bayesian_ar_detect"
app_prefix=${1}
input_regex_option=${2}
input_regex=${3}
f_time_template_option=${4}
f_time_template=${5}
t_axis_var_option=${6}
eval t_axis_var=${7}
t_units_option=${8}
t_units="${9}"
output_file_option=${10}
output_file="${11}"

# Second command
output_file_type=${12}
output_ref="${13}"

app_exec="${app_prefix}/${app_name}"
verifier_exec="${app_prefix}/teca_cartesian_mesh_diff"

${app_exec} ${input_regex_option} ${input_regex} \
    ${f_time_template_option} ${f_time_template} \
    ${t_axis_var_option} "${t_axis_var}" \
    ${t_units_option} "${t_units}" \
    ${output_file_option} ${output_file}

${verifier_exec} ${output_file_type} ${output_file} ${output_ref}

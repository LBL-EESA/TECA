#!/bin/bash

if [[ $# -ne 15 ]]
then
    echo "usage: test_deeplabv3p_ar_detect_app.sh" \
         "[app_prefix] [model_option] [model_file]" \
         "[input_regex_option] [input_regex] " \
         "[f_time_template_option] [f_time_template]" \
         "[t_axis_var_option] [t_axis_var]" \
         "[t_units_option] [t_axis_var]" \
         "[t_units_option] [t_units]" \
         "[output_file_option] [output_file]" \
         "[output_file_type] [output_ref]"
    exit 1
fi

# First command
app_name="teca_deeplabv3p_ar_detect"
app_prefix=${1}
model_option=${2}
model_file=${3}
input_regex_option=${4}
input_regex=${5}
f_time_template_option=${6}
f_time_template=${7}
t_axis_var_option=${8}
eval t_axis_var=${9}
t_units_option=${10}
t_units=${11}
output_file_option=${12}
output_file=${13}

# Second command
output_file_type=${14}
output_ref=${15}

app_exec="${app_prefix}/${app_name}"
verifier_exec="${app_prefix}/teca_cartesian_mesh_diff"

${app_exec} ${model_option} ${model_file} \
    ${input_regex_option} ${input_regex} \
    ${f_time_template_option} ${f_time_template} \
    ${t_axis_var_option} "${t_axis_var}" \
    ${t_units_option} "${t_units}" \
    ${output_file_option} ${output_file}

${verifier_exec} ${output_file_type} ${output_file} ${output_ref}

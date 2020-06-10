#!/usr/bin/env python3
import teca

def create_teca_pipeline(input_filenames = ["test_vertical_integral.nc"],
                         using_hybrid = True,
                         output_filename = "test.nc"):
    """Creates a TECA pipeline that tests teca_vertical_integral

    input:
    ------

        input_filenames  : a list containing file names to open
        
        using_hybrid     : flags whether to use exercise the hybrid coordinate
                           integration (does sigma coordinate otherwise)

        output_filenames : the output filename to which to write the result

    output:
    -------

        pipeline : the last algorithm in the TECA pipeline; running
                   pipeline.update() will run the pipeline.

        When run, the pipeline writes a netCDF file to disk, containing the 
        result of the integration.

    """
    # ***********************
    # Read the input dataset
    # ***********************
    cfr = teca.teca_cf_reader.New()
    cfr.set_file_names(input_filenames)
    # set coordinate names in the input dataset
    cfr.set_x_axis_variable("x")
    cfr.set_y_axis_variable("y")
    # TODO: how to deal with two vertical coordiantes: kz and kzi?
    cfr.set_z_axis_variable("kz")
    cfr.set_t_axis_variable("time")
    # turn off threading
    cfr.set_thread_pool_size(1)

    # **********************
    # Normalize coordinates
    # **********************
    coords = teca.teca_normalize_coordinates.New()
    coords.set_input_connection(cfr.get_output_port())

    # ***********************
    # Integrate `test_field`
    # ***********************
    vint = teca.teca_vertical_integral.New()
    vint.set_input_connection(coords.get_output_port())
    # set variables needed for integration
    vint.set_hybrid_a_variable("a_bnds")
    vint.set_hybrid_b_variable("b_bnds")
    #vint.set_sigma_variable("sigma_i")
    vint.set_surface_p_variable("ps")
    #vint.set_p_top_variable("p_top")
    vint.set_integration_variable("test_field")
    # set output metadata
    vint.set_output_variable_name("vint_test_field")
    vint.set_long_name("vertical integral of test_field")
    vint.set_units("none")
    # flag that we're integrating using the hybrid coordinate
    vint.set_using_hybrid(int(using_hybrid))
    vint.set_p_top_override_value(0.4940202)

    if False:
        class metadata_override(teca.teca_python_algorithm):

            def report(self, port, md_in):
                md_out = teca.teca_metadata(md_in[0])

                wext = md_out['whole_extent']
                ncells = (wext[1] - wext[0] + 1)* \
                         (wext[3] - wext[2] + 1)*(wext[5] - wext[4] + 1)

                new_var_atts = teca.teca_array_attributes(
                        teca.teca_double_array_code.get(),
                        teca.teca_array_attributes.point_centering,
                        int(ncells), 'unitless', 'vertical integral of -1/g',
                        'should be approximately equal to ps')
                        
                # put it in the array attributes
                try:
                    atts = md_out['attributes']
                except:
                    atts = teca_metadata()
                atts['vint_test_field'] = new_var_atts.to_metadata()
                md_out['attributes'] = atts
                return md_out

        mdi = metadata_override.New()
        mdi.set_input_connection(vint.get_output_port())

    # stub for looping over the singular time dimension
    exe = teca.teca_index_executive.New()
    exe.set_start_index(0)
    exe.set_end_index(0)

    # *********************
    # Write an output file
    # *********************
    wri = teca.teca_cf_writer.New()
    wri.set_file_name(output_filename)
    wri.set_executive(exe)
    wri.set_input_connection(vint.get_output_port())
    wri.set_point_arrays(["vint_test_field"])
    #wri.set_input_connection(coords.get_output_port())
    wri.set_thread_pool_size(1)


    return wri

# create the TECA pipeline
pipeline = create_teca_pipeline()

print(pipeline.update_metadata())

# run the pipeline
pipeline.update()

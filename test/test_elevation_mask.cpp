#include "teca_config.h"
#include "teca_cf_reader.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_normalize_coordinates.h"
#include "teca_elevation_mask.h"
#include "teca_indexed_dataset_cache.h"
#include "teca_cf_writer.h"
#include "teca_cf_reader.h"
#include "teca_multi_cf_reader.h"
#include "teca_dataset_diff.h"
#include "teca_file_util.h"
#include "teca_system_util.h"

#include "teca_index_executive.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"
#include "teca_mpi.h"

#include <vector>
#include <string>
#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    if (argc < 5)
    {
        if (rank == 0)
        {
            std::cerr << std::endl << "Usage error:" << std::endl
                << "test_elevation_mask [reader type (cf,mcf)] [mesh files]"
                   " [surface elev file] [out file] [first step] [last step]"
                << std::endl << std::endl;
        }
        return -1;
    }
    std::string reader_type = argv[1];
    std::string mesh_regex = argv[2];
    std::string elev_regex = argv[3];
    std::string baseline = argv[4];
    int first_step = (argc > 5 ? atoi(argv[5]) : 0);
    int last_step = (argc > 6 ? atoi(argv[6]) : -1);

    // mesh
    p_teca_algorithm mesh_reader;
    if (reader_type == "cf")
    {
        p_teca_cf_reader cfr = teca_cf_reader::New();
        cfr->set_z_axis_variable("plev");
        cfr->set_files_regex(mesh_regex);
        mesh_reader = cfr;
    }
    else if (reader_type == "mcf")
    {
        p_teca_multi_cf_reader cfr = teca_multi_cf_reader::New();
        cfr->set_z_axis_variable("plev");
        cfr->set_input_file(mesh_regex);
        mesh_reader = cfr;
    }

    p_teca_normalize_coordinates mesh_coords = teca_normalize_coordinates::New();
    mesh_coords->set_input_connection(mesh_reader->get_output_port());

    teca_metadata md = mesh_coords->update_metadata();

    // surface elevations
    p_teca_cf_reader elev_cfr = teca_cf_reader::New();
    elev_cfr->set_files_regex(elev_regex);
    elev_cfr->set_t_axis_variable("");

    p_teca_normalize_coordinates elev_coords = teca_normalize_coordinates::New();
    elev_coords->set_input_connection(elev_cfr->get_output_port());
    elev_coords->set_enable_periodic_shift_x(1);

    // regrid the surface elevations onto the mesh coordinates
    p_teca_cartesian_mesh_source elev_tgt = teca_cartesian_mesh_source::New();
    elev_tgt->set_spatial_bounds(md, false);
    elev_tgt->set_spatial_extents(md, false);
    elev_tgt->set_x_axis_variable(md);
    elev_tgt->set_y_axis_variable(md);
    elev_tgt->set_z_axis_variable(md);
    elev_tgt->set_t_axis_variable(md);
    elev_tgt->set_t_axis(md);

    p_teca_cartesian_mesh_regrid regrid = teca_cartesian_mesh_regrid::New();
    regrid->set_input_connection(0, elev_tgt->get_output_port());
    regrid->set_input_connection(1, elev_coords->get_output_port());

    p_teca_algorithm head = regrid;
#ifdef TECA_DEBUG
    p_teca_cartesian_mesh_writer vtkwr = teca_cartesian_mesh_writer::New();
    vtkwr->set_input_connection(regrid->get_output_port());
    vtkwr->set_file_name("out_%t%.vtk");
    head = vtkwr;
#endif

    p_teca_indexed_dataset_cache elev_cache = teca_indexed_dataset_cache::New();
    elev_cache->set_input_connection(head->get_output_port());
    elev_cache->set_max_cache_size(1);

    // mask
    p_teca_elevation_mask mask = teca_elevation_mask::New();
    mask->set_input_connection(0, mesh_coords->get_output_port());
    mask->set_input_connection(1, elev_cache->get_output_port());
    mask->set_mesh_height_variable("zg");
    mask->set_surface_elevation_variable("z");
    mask->set_mask_variables({"hus_valid","ua_valid","va_valid"});

    p_teca_index_executive rex = teca_index_executive::New();
    rex->set_verbose(1);

    std::vector<std::string> arrays({"zg", "hus", "hus_valid",
            "ua", "ua_valid", "va", "va_valid"});


    // run the test or generate the baseline
    bool do_test = true;
    teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);

    // find files named by a regex.
    std::string tmp_baseline = baseline + ".*\\.nc$";
    std::vector<std::string> test_files;
    std::string regex = teca_file_util::filename(tmp_baseline);
    std::string tmp_path = teca_file_util::path(tmp_baseline);
    if (do_test && !teca_file_util::locate_files(tmp_path, regex, test_files))
    {
        std::cerr << "running the test ..." << std::endl;

        p_teca_cf_reader cfr = teca_cf_reader::New();
        cfr->set_files_regex(baseline);
        cfr->set_z_axis_variable("plev");

        rex->set_arrays(arrays);

        p_teca_dataset_diff diff = teca_dataset_diff::New();
        diff->set_input_connection(0, cfr->get_output_port());
        diff->set_input_connection(1, mask->get_output_port());
        diff->set_executive(rex);
        diff->set_verbose(1);

        diff->update();
    }
    else
    {
        std::cerr << "writing the baseline ..." << std::endl;
        tmp_baseline = baseline + ".nc";

        p_teca_cf_writer cmw = teca_cf_writer::New();
        cmw->set_input_connection(mask->get_output_port());
        cmw->set_thread_pool_size(1);
        cmw->set_file_name(tmp_baseline);
        cmw->set_steps_per_file(10000);
        cmw->set_point_arrays(arrays);

        cmw->set_first_step(first_step);
        cmw->set_last_step(last_step);
        cmw->set_verbose(1);

        cmw->set_executive(rex);
        cmw->update();
    }

    return 0;
}
